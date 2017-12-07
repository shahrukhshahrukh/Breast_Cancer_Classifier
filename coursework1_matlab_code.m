clear;

data = importdata('brw_data.txt');
inputData = data(:,2:10);
targetData = data(:,11);

dataamount  = 500;
paramepochs = 10;
paramgoal = 0.11;
parammaxfail = 5;
hiddenlayers = 10;
result = [];

trd12 = inputData(1:dataamount/2,:);
trd14 = inputData((size(inputData)-dataamount/2)+1:size(inputData),:);
trr12 = targetData(1:dataamount/2,:);
trr14 = targetData((size(targetData)-dataamount/2)+1:size(targetData),:);

trainingInputData = cat(1,trd12,trd14);
trainingTargetData = cat(1,trr12,trr14);
testingData = inputData((dataamount/2)+1:(size(inputData)-dataamount/2),:);
testingTargetData = targetData((dataamount/2)+1:(size(targetData)-dataamount/2),:);

seed = RandStream('mt19937ar', 'seed', 1);
RandStream.setGlobalStream(seed);

net = newff(trainingInputData',trainingTargetData',hiddenlayers, {'tansig' 'tansig'}, 'trainr', 'learngd', 'mse');
net.trainParam.epochs = paramepochs;
net.trainParam.goal = paramgoal;
net.trainParam.max_fail = parammaxfail;
net = train(net,trainingInputData',trainingTargetData');
netOutput = net(testingData');

matchCounter = 0;
netOutput = netOutput';
for i=1:size(netOutput)
    if(netOutput(i)<=2.5)
        netOutput(i)=2;
    else
        netOutput(i)=4;
    end
    if(testingTargetData(i)==netOutput(i))
        matchCounter = matchCounter+1;
    end
end

percentagePerformance = (matchCounter/size(netOutput,1))*100;
disp(dataamount);
disp(paramepochs);
disp(paramgoal);
disp(parammaxfail);
disp(percentagePerformance);
disp('-------------------------------');

res = [];
res = cat(2, percentagePerformance, res);
res = cat(2, paramgoal, res);
res = cat(2, paramepochs, res);
res = cat(2, parammaxfail, res);
res = cat(2, dataamount, res);
result = cat(1, res, result);