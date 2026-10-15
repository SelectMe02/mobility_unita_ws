"""Shared date-named folder for one log:=true run.

The recorder creates log/run_YYYYmmdd_HHMMSS and announces it on the ROS
parameter server under the launch's session token. Other nodes of the same
launch wait for that token, so a folder from an earlier run is never reused.
"""
from datetime import datetime
from pathlib import Path

DIRECTORY_PARAM = '/driving_log/directory'
SESSION_PARAM = '/driving_log/session'


def new_run_directory(root, now=None):
    root = Path(root).expanduser().resolve()
    name = 'run_' + (now or datetime.now()).strftime('%Y%m%d_%H%M%S')
    candidate, suffix = root / name, 2
    while candidate.exists():
        candidate, suffix = root / '{}_{}'.format(name, suffix), suffix + 1
    candidate.mkdir(parents=True)
    return candidate


def announce(rospy, directory, session):
    rospy.set_param(DIRECTORY_PARAM, str(directory))
    rospy.set_param(SESSION_PARAM, str(session))


def announced_directory(rospy, session):
    """The directory announced for this launch's session, or None yet."""
    if not session or rospy.get_param(SESSION_PARAM, None) != str(session):
        return None
    directory = rospy.get_param(DIRECTORY_PARAM, None)
    return Path(directory) if directory else None
