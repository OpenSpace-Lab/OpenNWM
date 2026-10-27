"""Session gait selection and zero-velocity normal stops; ROS supplied by the base."""


class SessionGaitControl:
    def __init__(self, *args, **kwargs):
        self._static_gait_ready = False
        self._static_gait_type = None
        self._settling_at_zero = False
        super().__init__(*args, **kwargs)

    def invalidate_gait(self):
        with self.lock:
            self._static_gait_ready = False
            self._static_gait_type = None

    def static_walk(self):
        with self.lock:
            if self.fault:
                raise RuntimeError(self.fault)
            if not self.execute:
                raise RuntimeError('Read-only adapter cannot select a gait')
            sport = getattr(self, 'sport', None)
            current = getattr(sport[1], 'gait_type', None) if sport else None
            if self._static_gait_ready and current == self._static_gait_type:
                return False
            self._static_gait_ready = False
            super().static_walk()
            return True

    def confirm_static_gait(self):
        # Called only after the post-switch stability check succeeds.
        with self.lock:
            if self.fault:
                raise RuntimeError(self.fault)
            sport = getattr(self, 'sport', None)
            self._static_gait_type = getattr(sport[1], 'gait_type', None) if sport else None
            self._static_gait_ready = True

    def pause_motion(self):
        # Keep the regular command watchdog active throughout settling. The
        # read loop below refreshes zero commands; a stalled loop still stops.
        with self.lock:
            self.move(0., 0., 0.)
            self._settling_at_zero = True

    def read_pose(self):
        pose = super().read_pose()
        with self.lock:
            if self._settling_at_zero:
                self.move(0., 0., 0.)
        return pose

    def release_motion_control(self):
        """After measured settling, return RC control without a Sport stop RPC."""
        with self.lock:
            if self.fault:
                raise RuntimeError(self.fault)
            if self.has_avoidance_control:
                if not self._settling_at_zero:
                    raise RuntimeError('Cannot release normal motion before zero-velocity settling')
                self.publish_avoidance(1004, {'is_remote_commands_from_api': False}, noreply=True)
                self.has_avoidance_control = False
            self.active = False
            self._settling_at_zero = False

    def stop(self):
        # Faults, reserve limits, operator exit and watchdog still use the
        # original emergency stop path, including Sport StopMove and release.
        with self.lock:
            self._settling_at_zero = False
            self.invalidate_gait()
            return super().stop()
