def card(name: str, **props) -> dict:
    return {"type": "card", "card": name, "props": props}


def stage(name: str) -> dict:
    return {"type": "stage_change", "stage": name}


def error(code: str, message: str, field: str | None = None) -> dict:
    return {"type": "error", "code": code, "message": message, "field": field}


def text_delta(text: str) -> dict:
    return {"type": "text_delta", "text": text}
