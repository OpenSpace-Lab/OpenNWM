#!/usr/bin/env python3
"""Preflight or rename a ModelScope dataset without copying its contents.

The default is read-only. Pass --apply to rename, and --expected-id to pin the
numeric repository identity and recognize a completed rename on subsequent runs.
Credentials come from MODELSCOPE_API_TOKEN or an existing ModelScope SDK login;
the script never logs credentials or writes the SDK credential store.

The official website implements this endpoint (inspected 2026-10-16):
https://g.alicdn.com/sail-web/maas/2.13.166/umi.js
https://g.alicdn.com/sail-web/maas/2.13.166/p__datasetsDetail__Settings__index.async.js
The website warns that renaming invalidates old URLs, including Git URLs.
"""

import argparse
import json
import os
import sys
import time
from urllib.parse import quote, unquote, urlparse

import requests


class RenameError(RuntimeError):
    """An error whose message is safe to display without credentials."""


def parse_repo_id(value):
    parts = value.split('/')
    if len(parts) != 2 or any(not part or part in ('.', '..') for part in parts):
        raise argparse.ArgumentTypeError('Use the owner/dataset repository format.')
    if any(character.isspace() for character in value):
        raise argparse.ArgumentTypeError('Repository names cannot contain whitespace.')
    return value


def build_session(endpoint, use_env_proxy=False):
    """Reuse SDK credentials; an explicit environment token takes precedence."""
    session = requests.Session()
    session.trust_env = use_env_proxy
    session.headers.update({'Accept': 'application/json'})
    host = urlparse(endpoint).hostname
    token = os.environ.get('MODELSCOPE_API_TOKEN', '').strip()
    cookies = None
    if not token:
        try:
            from modelscope_hub.config import HubConfig
        except ImportError:
            try:
                from modelscope.hub.api import ModelScopeConfig
            except ImportError:
                pass
            else:
                cookies = ModelScopeConfig.get_cookies()
        else:
            config = HubConfig(endpoint=endpoint)
            cookies = config.load_cookies()
            token = config.token or ''
    if cookies:
        for cookie in cookies:
            domain = cookie.domain.lstrip('.')
            if not cookie.is_expired() and (host == domain or host.endswith('.' + domain)):
                session.cookies.set_cookie(cookie)
    # This mirrors the SDK legacy transport: both bearer and m_session_id are
    # accepted, depending on the endpoint. Preserve server-issued session cookies.
    if token:
        session.headers['Authorization'] = 'Bearer ' + token
        if not any(cookie.name == 'm_session_id' for cookie in session.cookies):
            session.cookies.set('m_session_id', token, domain=host, path='/', secure=True)
    for cookie in session.cookies:
        if cookie.name == 'csrf_token':
            session.headers['X-CSRF-TOKEN'] = unquote(cookie.value)
    authenticated = bool(token or any(cookie.name == 'm_session_id' for cookie in session.cookies))
    return session, authenticated


class DatasetClient:
    def __init__(self, session, endpoint, timeout=30):
        self.session = session
        self.endpoint = endpoint.rstrip('/')
        self.timeout = timeout

    def request(self, method, path, payload=None, missing_ok=False):
        # Do not automatically retry POST: a lost response could follow a
        # successful rename. A later invocation verifies the numeric identity.
        try:
            response = self.session.request(
                method, self.endpoint + path, json=payload,
                timeout=(10, self.timeout), allow_redirects=False,
            )
        except requests.RequestException:
            raise RenameError(
                'Network request failed. If --apply was used, rerun the read-only '
                'preflight with --expected-id to determine whether the rename completed.'
            ) from None
        if missing_ok and response.status_code == 404:
            return None
        if response.status_code in (401, 403):
            raise RenameError('Authentication or permission rejected. Refresh your ModelScope login.')
        if not 200 <= response.status_code < 300:
            raise RenameError(f'ModelScope returned HTTP {response.status_code}; no response body logged.')
        try:
            body = response.json()
        except ValueError:
            raise RenameError('ModelScope returned a non-JSON response.') from None
        if not isinstance(body, dict):
            raise RenameError('ModelScope returned an unexpected JSON response.')
        code = body.get('Code')
        if body.get('Success') is False or code not in (None, 200, '200'):
            # Only numeric codes are printed; server error messages can echo inputs.
            display_code = str(code) if str(code).isdigit() else 'unknown'
            raise RenameError(f'ModelScope rejected the request (code {display_code}).')
        return body

    def get(self, repo_id):
        path = '/api/v1/datasets/' + '/'.join(quote(part, safe='') for part in repo_id.split('/'))
        body = self.request('GET', path, missing_ok=True)
        if body is None:
            return None
        data = body.get('Data')
        if not isinstance(data, dict):
            raise RenameError('Dataset metadata is missing from the response.')
        try:
            numeric_id = int(data['Id'])
        except (KeyError, TypeError, ValueError):
            raise RenameError('Dataset metadata does not contain a numeric Id.') from None
        return {'repo_id': f"{data.get('Owner') or data.get('Namespace')}/{data.get('Name')}",
                'id': numeric_id}

    def rename(self, source, target):
        old_owner, old_name = source.split('/')
        new_owner, new_name = target.split('/')
        self.request('POST', '/api/v1/datasets/rename/repo', {
            'OldNamespace': old_owner, 'OldName': old_name,
            'NewNamespace': new_owner, 'NewName': new_name,
        })


def run(client, source, target, expected_id=None, apply=False, authenticated=False):
    before = client.get(source)
    destination = client.get(target)
    if before is None:
        if destination is not None and expected_id == destination['id'] and destination['repo_id'] == target:
            return {'status': 'already_renamed', 'source': source, 'target': target,
                    'id': expected_id, 'changed': False}
        if destination is not None and expected_id is None:
            raise RenameError('Source is absent and target exists. Supply the original numeric --expected-id to verify identity.')
        raise RenameError('Source is absent; a completed rename with the expected identity could not be verified.')
    if before['repo_id'] != source:
        raise RenameError('Source metadata resolves to a different repository name; inspect the remote state.')
    if expected_id is not None and before['id'] != expected_id:
        raise RenameError('Source numeric Id differs from --expected-id. No rename performed.')
    if destination is not None:
        raise RenameError('Target already exists. No rename performed.')
    result = {'status': 'ready', 'source': source, 'target': target, 'id': before['id'],
              'credentials_available': authenticated, 'changed': False,
              'old_urls': 'ModelScope states that old URLs, including Git URLs, become invalid.'}
    if not apply:
        return result
    if not authenticated:
        raise RenameError('No current SDK credentials or MODELSCOPE_API_TOKEN found. Refresh your ModelScope login.')
    client.rename(source, target)
    # Metadata propagation may take a few seconds. Poll GET only, never retry POST.
    for attempt in range(5):
        destination = client.get(target)
        if destination is not None:
            if destination['id'] != before['id'] or destination['repo_id'] != target:
                raise RenameError('Rename submitted, but the target identity differs. Inspect the remote repository before proceeding.')
            result.update(status='renamed', changed=True)
            return result
        if attempt < 4:
            time.sleep(2)
    raise RenameError('Rename submitted but not yet verified. Rerun preflight with the original --expected-id; do not assume failure.')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--source', type=parse_repo_id, default='LittleBoss/LAM-Data')
    parser.add_argument('--target', type=parse_repo_id, default='LittleBoss/NavAnywhere')
    parser.add_argument('--expected-id', type=int, help='Original numeric dataset Id; required to verify an already completed rename.')
    parser.add_argument('--apply', action='store_true', help='Perform the rename after preflight.')
    parser.add_argument('--endpoint', choices=['https://modelscope.cn', 'https://modelscope.ai'],
                        default=os.environ.get('MODELSCOPE_ENDPOINT', 'https://modelscope.cn').rstrip('/'))
    parser.add_argument('--use-env-proxy', action='store_true', help='Honor HTTP(S)_PROXY; the default connects directly.')
    parser.add_argument('--timeout', type=int, default=30, help='Response timeout in seconds (default: 30).')
    args = parser.parse_args()
    if args.source == args.target:
        parser.error('--source and --target must differ.')
    if args.source.split('/')[0] != args.target.split('/')[0]:
        parser.error('This helper only renames within the same owner; ownership transfer is outside its scope.')
    if args.timeout <= 0 or args.expected_id is not None and args.expected_id <= 0:
        parser.error('--timeout and --expected-id must be positive.')
    if args.endpoint not in ('https://modelscope.cn', 'https://modelscope.ai'):
        parser.error('MODELSCOPE_ENDPOINT must be https://modelscope.cn or https://modelscope.ai.')
    session = None
    try:
        session, authenticated = build_session(args.endpoint, args.use_env_proxy)
        result = run(DatasetClient(session, args.endpoint, args.timeout), args.source,
                     args.target, args.expected_id, args.apply, authenticated)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except RenameError as error:
        print(f'ERROR: {error}', file=sys.stderr)
        return 1
    except Exception as error:
        # Never dump a traceback or exception repr that could include auth data.
        print(f'ERROR: {type(error).__name__}; details suppressed to protect credentials.', file=sys.stderr)
        return 1
    finally:
        if session is not None:
            session.close()


if __name__ == '__main__':
    sys.exit(main())
