from fastapi import APIRouter, HTTPException
import pandas as pd
from service.indicator import index_generate  # 確保這個導入路徑與你的項目結構匹配
from model.indicator import IndexRequest  # 假設IndexRequest模型在model.py文件中
import logging

router = APIRouter()


@router.post("/generate-indexes/")
async def generate_indexes(request: IndexRequest):
    try:
        # 將請求中的價格數據轉換成DataFrame
        df = pd.DataFrame(request.price_data)
        # rename column date to time
        df.rename(columns={"date": "time"}, inplace=True)

        # 調用 index_generate 函數生成指標
        df_w_index = index_generate(df, request.required_ta)

        # 將結果DataFrame轉換為字典列表以返回
        return df_w_index.reset_index().to_dict("records")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
