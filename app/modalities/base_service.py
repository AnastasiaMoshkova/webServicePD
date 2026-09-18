from typing import Any, Dict


class BaseModalityService:
    def predict_binary(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError("predict_binary is not implemented")

    def upload(self, file_name: str, content: bytes) -> Dict[str, Any]:
        raise NotImplementedError("upload is not implemented")

    def raw_data_processing(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError("raw_data_processing is not implemented")

    def insert_in_db(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError("insert_in_db is not implemented")

