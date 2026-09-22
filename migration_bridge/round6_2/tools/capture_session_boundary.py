"""Fail before inference when a fresh capture session contains prior forwards."""


def require_empty_capture_session(path):
    if not path.is_dir():
        raise ValueError('CAPTURE_DIRECTORY_NOT_READY')
    prior = next(path.rglob('Pass*.pt'), None)
    incomplete = next(path.rglob('*.incomplete'), None)
    if prior is not None or incomplete is not None:
        raise ValueError('CAPTURE_SESSION_NOT_EMPTY')
