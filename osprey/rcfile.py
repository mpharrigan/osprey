import os
import yaml

__all__ = ['load_rcfile', 'CURDIR_RC_PATH', 'USER_RC_PATH']


CURDIR_RC_PATH = os.path.abspath('.ospreyrc')
USER_RC_PATH = os.path.abspath(os.path.expanduser('~/.ospreyrc'))


def load_rcfile():
    def get_rc_path():
        path = os.getenv('OSPREYRC')
        if path == ' ':
            return None
        if path:
            return path
        for path in (CURDIR_RC_PATH, USER_RC_PATH):
            if os.path.isfile(path):
                return path
        return None

    path = get_rc_path()
    if not path:
        return {}
    print('Loading %s...' % path)

    with open(path) as f:
        return yaml.load(f) or {}