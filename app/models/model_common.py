from pydantic import BaseModel
from typing import Optional


class ResCommonVo(BaseModel):
    result_code: Optional[str]
    result_message: Optional[str]
