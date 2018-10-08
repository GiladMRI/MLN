addpath('./MATLAB_Utils');
P2DFPBase='/home/a/Downloads/MLN-master/P2DF/';
TFBase='/home/a/Downloads/MLN-master/';
LUserName='a';

% system(['sudo chmod +X ' TFBase 'RunTFForMatlab.sh']);
% /media/a/f38a5baa-d293-4a00-9f21-ea97f318f647/home/a/
Accs=1.4:0.05:3;
WhichAccs=[1 12 19 26 33];
% The acceleration factors we ran on are: 1.4 1.95 2.3 2.65 3.0
% These numbers are for "1D" acceleration according to BART/poisson module
% And result in the higher 2D acceleration factors described in the paper
CurAccIdx=3; % 3 is 
WhichAcc=WhichAccs(CurAccIdx);
CurAccStr=num2str(Accs(WhichAcc),'%.2f');
disp(['Currend 1D acceleration: ' CurAccStr]);
% Training:
P2DFP=[P2DFPBase CurAccStr filesep];
%% Now prepare the data for the net
load('Benchmark/SamplingMasks.mat','MskM','WhichAccs');
load('Benchmark/TestImages.mat','TIs');
% TIs is a cell array containing the benchmark images.
% Images may be added (up to 16 in every batch)

nTIs=numel(TIs);
CurMask=MskM(:,:,WhichAcc);
ActAcc=numel(CurMask)/sum(CurMask(:));
fgmontage(CurMask);
xlabel(['"1D" Acc: ' num2str(Accs(WhichAcc),'%.2f') '. Actual undersampling: ' num2str(ActAcc,'%.2f')]);
%%
[F1 F2]=find(CurMask);
nTraj=numel(F1);

clear MDataT
Traj=[F1.';F2.']-100;
for t=1:nTIs
    ImN=double(TIs{t}).*MskS;
    ImN=ImN*0.3/grmss(ImN);
    NormalizedBenchmarkImages(:,:,t)=abs(ImN);
    ImWithSens=ImN.*Sens;
    FData=fft2cg(ImWithSens);
    for i=1:nTraj
        MDataT(i,:,t)=FData(F1(i),F2(i),:);
    end
end
CurAccStr=num2str(Accs(WhichAcc),'%.2f');
P2DFP=[P2DFPBase CurAccStr filesep];

%
clear CurIDataVR
for t=1:nTIs
    tmp=MDataT(:,:,t); % squeeze(DataPCC);
    RealDataFac=2;
    CurIDataV=Row(tmp)*RealDataFac;
    CurIDataVR(t,:)=[real(CurIDataV) imag(CurIDataV)];
end
% Data=repmat(single(CurIDataVR),[16 1]);
Data=CurIDataVR;
Data(16,1)=0;
RealDataFN=[P2DFP 'RealDataForNN01.mat'];
save(RealDataFN,'Data');
disp('Saved Data for MLN');
fgmontage(NormalizedBenchmarkImages)
%% Prepare parameters
D=dir([P2DFPBase CurAccStr filesep]);
D=D([D.isdir]);
D=D(strhas({D.name},'train'));
St=getParamsStructFromFN([P2DFPBase CurAccStr filesep D(1).name filesep]);
St.nTraj=nTraj;
St.DataH=nTraj*8*2;
St.SessionNameBase=['RegridTry3C2_7TS_P2DF_' CurAccStr];
St.BaseTSDataP=P2DFP;
St.BaseNUFTDataP=P2DFP;
St.RealDataFN=[P2DFP 'RealDataForNN.mat'];
St.DatasetMatFN=[TFBase 'HCPData_256x256_int16.mat'];
St.nToLoad=10000;
St.LoadAndRunOnData=1;
D=dir([P2DFP 'RegridTry3C2_7TS_P2DF_' CurAccStr '*checkpoint*']);
St.LoadAndRunOnData_checkpointP=[P2DFP D.name];
St.LoadAndRunOnData_Prefix=[P2DFP 'RealDataForNN'];
St.LoadAndRunOnData_OutP=P2DFP;
St.HowManyToRun=1;
Txt=gStruct2txt(St,[TFBase 'Params.txt']);
disp('Prepared Params');
%% Call tensorflow
system(['sudo -H -u ' LUserName ' ' TFBase 'RunTFForMatlab.sh']);
%% Collect the results
tmp=load([P2DFP 'OnRealData01.mat']);
Recond=permute(tmp.x(:,:,:,1)+1i*tmp.x(:,:,:,2),[2 3 1]);
fgmontage(Recond(:,:,1:nTIs));
title('Reconstructed images');
xlabel(['"1D" Acc: ' num2str(Accs(WhichAcc),'%.2f') '. Actual undersampling: ' num2str(ActAcc,'%.2f')]);