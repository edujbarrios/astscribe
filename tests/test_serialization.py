import json

from astscribe import analyze


def test_result_to_dict_is_json_serializable() -> None:
    result = analyze("model.eval()")
    json.dumps(result.to_dict())
