# Language Modeling

## dataset.py
모델에 데이터를 제공하기 위해 Shakespeare dataset을 활용 
## model.py
`model.py`에서 바닐라 RNN 및 LSTM 모델을 구현합니다. 파일에 주석으로 몇 가지 지침이 제공됩니다. 모델 성능 향상에 도움이 된다면 원하는 만큼 레이어를 쌓아보세요.
## main.py
모델 학습을 위한 `main.py`를 작성합니다. 여기서는 학습 및 검증 데이터 세트의 평균 손실 값을 사용하여 학습 과정을 모니터링
The models were trained using the Adam optimizer and cross-entropy loss. We trained both models for 10 epochs and monitored the training and validation loss.
검증 데이터 세트의 손실값을 기준으로 바닐라 RNN과 LSTM의 언어 생성 성능을 비교

/Epoch 1, Train Loss RNN: 1.7380, Val Loss RNN: 1.5312
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

Results
The LSTM model showed better performance compared to the vanilla RNN model, achieving a lower validation loss.

Text Generation
We used the trained LSTM model to generate text samples starting from different seed characters.

## generate.py
temperature parameter *T* 
문자를 생성할 때 두 개의 temperature parameter를 비교 결과
temperature parameter가 높을 수록 다양하고 창의적인 텍스트를 생성할 수 있었지만 더 무의미하거나 덜 일관성 있는 시퀀스를 포함되는 아래 결과로 확인
- High Temperature (e.g., 0.8)

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

- Low Temperature (e.g., 0.3)

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
