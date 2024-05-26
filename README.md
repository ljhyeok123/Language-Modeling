# Language Modeling

## dataset.py
모델에 데이터를 제공하기 위해 Shakespeare dataset을 활용 
`shakespeare.txt`을 기반으로 데이터를 tenser 형식으로 변환하여 `shakespeare_train.txt`으로 저장
## model.py
`model.py`에서 바닐라 RNN 및 LSTM 모델을 구현
## main.py
모델 학습을 위해 `main.py`에서 RNN과 LSTM model의 parameter를 다음과 같이 정의

    batch_size = 64
    hidden_dim = 64
    num_layers = 3


`best_rnn_model.pth`와 `best_lstm_model.pth`파일은 각각 RNN과 LSTM 모델의 학습된 weight와 parameter를 저장한 파일입니다. 이 파일들은 `main.py` 스크립트에서 모델 학습 후, validation 데이터에서 가장 좋은 성능을 보인 모델을 저장

모델을 training할 때 최적화를 위해 Adam optimizer와 cross-entropy loss을 사용
CrossEntropyLoss는 다중 클래스 분류 문제에 적합하며 Adam는 SGD와 비교해 수렴 속도가 빠르고 learning rate을 자동으로 조정해 주기 때문에 hyperparmeter 튜닝이 비교적 용이한 장점이 있음

두 모델을 10회에 걸쳐 훈련하고 training 및 validation loss을 모니터링검증 데이터 세트의 손실값을 기준으로 바닐라 RNN과 LSTM의 언어 생성 성능을 비교

결과

LSTM 모델은 validation loss도 더 낮으며 RNN 모델에 비해 더 나은 성능을 보임

    Epoch 1, Train Loss RNN: 1.7380, Val Loss RNN: 1.5312
    Epoch 2, Train Loss RNN: 1.4868, Val Loss RNN: 1.4608
    Epoch 3, Train Loss RNN: 1.4399, Val Loss RNN: 1.4299
    Epoch 4, Train Loss RNN: 1.4159, Val Loss RNN: 1.4104
    Epoch 5, Train Loss RNN: 1.4007, Val Loss RNN: 1.3999
    Epoch 6, Train Loss RNN: 1.3899, Val Loss RNN: 1.3908
    Epoch 7, Train Loss RNN: 1.3827, Val Loss RNN: 1.3838
    Epoch 8, Train Loss RNN: 1.3768, Val Loss RNN: 1.3786
    Epoch 9, Train Loss RNN: 1.3723, Val Loss RNN: 1.3741
    Epoch 10, Train Loss RNN: 1.3687, Val Loss RNN: 1.3693
    
    Epoch 1, Train Loss LSTM: 1.9473, Val Loss LSTM: 1.5975
    Epoch 2, Train Loss LSTM: 1.5094, Val Loss LSTM: 1.4489
    Epoch 3, Train Loss LSTM: 1.4058, Val Loss LSTM: 1.3760
    Epoch 4, Train Loss LSTM: 1.3469, Val Loss LSTM: 1.3290
    Epoch 5, Train Loss LSTM: 1.3057, Val Loss LSTM: 1.2943
    Epoch 6, Train Loss LSTM: 1.2738, Val Loss LSTM: 1.2645
    Epoch 7, Train Loss LSTM: 1.2476, Val Loss LSTM: 1.2431
    Epoch 8, Train Loss LSTM: 1.2255, Val Loss LSTM: 1.2212
    Epoch 9, Train Loss LSTM: 1.2064, Val Loss LSTM: 1.2064
    Epoch 10, Train Loss LSTM: 1.1895, Val Loss LSTM: 1.1878

## generate.py
`generate.py`에서 학습된 LSTM 모델을 사용하여 텍스트를 생성, seed 문자를 사용하여 지정된 길이의 시퀀스를 생성하고 생성된 텍스트를 출력
문자를 생성할 때 두 개의 temperature parameter를 비교 결과

결과

temperature parameter가 높을 수록 다양하고 창의적인 텍스트를 생성할 수 있었지만 더 무의미하거나 덜 일관성 있는 시퀀스를 포함되는 아래 결과로 확인

High Temperature (e.g., 0.8)
    
    Seed: "Once "
    Once zealse and the while:
    The gods close to the time of your honour.--than me well.
    Second Citizen:
    Bei
    Seed: "When "
    When zealous hearts of the cortress
    Affareturn me, would be says he did
    I cannot hope, and so in the acco
    Seed: "While "
    While zook with such above me put in I have mine
    His husband fair work in my love of your poor lady of the
    Seed: "Although "
    Although zealing up the blood to blowd
    When I may to the reperity, that
    Methous the voice me for your hope,
    I
    Seed: "However "
    However zearl in the world
    Than I cannot be sunder ladies, the greater both
    A fellow of the last in a person

Low Temperature (e.g., 0.3)

    Seed: "Once "
    Once zealous speaks that the same the state and
    To see the people and the tribunes, the gods be the shado
    Seed: "When "
    When zeal is but the world of 'twixt the seat,
    When he was young, my lord, and there's not a hand.
    SICIN
    Seed: "While "
    While zealous singly son, as I say you that they
    Which will one to be so bad that he's a revenge of York,
    Seed: "Although "
    Although zealse the seat of a man,
    I'll plain and hear me she the mayor to your grace,
    The country's good wor
    Seed: "However "
    However zealous subjects the seat
    Are so still and the deep stand that he is the seat, I had he shall know t
