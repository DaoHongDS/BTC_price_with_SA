# BTC_price_with_SA

Do you own bitcoin (BTC)? Do you want to know the trend of BTC price in near future? I do.

And in this project, I have implemented the best model in the paper [Predicting the Price of Bitcoin Using Sentiment-Enriched Time Series Forecasting](https://www.mdpi.com/2504-2289/7/3/137)

# Data Processing

![image](https://drive.google.com/uc?export=view&id=1ekxoaxDdCWZqLqvlO43RfhvZ0u5ptbdA "BTC price data pipeline and how to feed it to forecasting model")

# Source code

```bash
├── 2_Swin_LSTM
│   ├── checkpoint
│   ├── model_swin_lstm
│   ├── Swin_npy
│   ├── Swin_LSTM_att.ipynb
├── 2_ViT_LSTM
│   ├── checkpoint
│   ├── model_trained_Vit_LSTM_att
│   ├── ViT_npy_preprocess
│   ├── ViT_LSTM_att.ipynb
├── 3_Yolo4_RNN
│   ├── checkpoint
│   ├── trained_model_Yolo_RNN_conv116
│   ├── Yolo4_conv116_npy_512_169
│   ├── Yolo4_conv116_npy_512_169.ipynb
├── 3_Yolo4_Xception_RNN
│   ├── checkpoint
│   ├── xception_npy
│   ├── trained_model_xception
│   ├── YoloCV2_Xception_LSTM.ipynb
├── 4_Swin_Trans
│   ├── model_swin_trans
│   ├── Swin_npy
│   ├── Swin_Trans.ipynb
├── 4_ViT_Trans
│   ├── model_ViT_Trans
│   ├── ViT_npy
│   ├── ViT_Trans.ipynb
├── 5_Yolo4_Trans
│   ├── model_Yolo4_Trans_conv116
│   ├── Yolo4_conv116_npy
│   ├── Yolo4_Trans_conv116.ipynb
├── 5_Yolo4_Xception_Trans
│   ├── model_Yolo4_Xception_Trans
│   ├── Yolo4_Xception_npy
│   ├── Yolo4_Xception_Trans.ipynb
```
