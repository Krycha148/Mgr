% Zadeklarowanie wartości czasu od 0 do 1000 co 2
clear all;
czas = 0:1:100;
wart = sin(czas(1,:))+2*cos(2*czas(1,:));

%Podzial
N=floor(0.9*length(wart));
dataTrain = wart(1:N+1);
dataTest = wart(N+1:end);

%Std
mean = mean(dataTrain);
std = std(dataTrain);
dataTrainStd = (dataTrain-mean)/std;

%Podzial
XTrain = dataTrainStd(1:end-1);
YTrain = dataTrainStd(2:end);

%LSTM
numFeatures = 1;
numResponses = 1;
numHiddenUnites = 200;

layers = [ ...
    sequenceInputLayer(numFeatures)
    lstmLayer(numHiddenUnites)    
    fullyConnectedLayer(numResponses)
    regressionLayer];
options = trainingOptions('adam',...
    'MaxEpochs',100, ...
    'GradientThreshold',1,...
    'InitialLearnRate',0.002,...
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropPeriod',50,...
    'LearnRateDropFactor',0.2,...
    'Verbose',0,...
    'Plots','training-progress');
%Uczenie sieci
net = trainNetwork(XTrain,YTrain,layers,options);

%Standaryzacja danych wejściowych na bazie zbioru treningowego
dataTestStd = (dataTest - mean)/std;
%Przygotowanie wejścia
XTest = dataTestStd(1:end-1);
numTimeStepsTest = numel(XTest);

%Podejscie 1 
YPred= predict(net,XTest);

% Podejscie 2 
net = predictAndUpdateState(net,XTrain);
[net,YPred2]= predictAndUpdateState(net,YTrain(end));
for i=2:numTimeStepsTest
    [net,YPred2(1,i)]=predictAndUpdateState(net,YPred(1,i-1),'ExecutionEnvironment','cpu');
end

YPred = std*YPred+mean;
YTest = dataTest(2:end);
figure(2)
grid on;
plot(dataTrain(50:end-1),'b-')
hold on;
idx=N:(N+numTimeStepsTest);
idx=idx-49;
plot(idx,[wart(N) YPred],'r-')
hold on;
plot(idx,[wart(N) YTest],'g--')
hold off;
xlabel("Time")
ylabel("Values")
title("Forecast")
legend(["Trening" "Pred" "Labels"])













