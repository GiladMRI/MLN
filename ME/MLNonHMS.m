B0TS=load('RealData/B0TS.mat');
RealDataForNN=load('RealData/RealDataForNN.mat');
SensCC1=load('RealData/SensCC1.mat');
Traj=load('RealData/Traj.mat');
TrajForNUFT=load('RealData/TrajForNUFT.mat');
A=load('RealData/OnRealData01.mat');
OnRealData01=permute(A.x(:,:,:,1)+1i*A.x(:,:,:,2),[2 3 1]);
A=load('RealData/Sli6_r01.mat');
Sli6_r01=A.Data;
A=load('RealData/Sli6_r02.mat');
Sli6_r02=A.Data;
disp('Read RealData stuff');
%% Used:
% B0Data=scipy.io.loadmat(BaseTSDataP + 'B0TS.mat')
% TSBF=B0Data['TSBF']
% TSC=B0Data['TSC']
% 
% SensCC=scipy.io.loadmat(BaseTSDataP + 'SensCC1.mat')
% Sens=SensCC['SensCC']
% SensMsk=SensCC['SensMsk']
% 
% NUFTData=scipy.io.loadmat(BaseNUFTDataP + 'TrajForNUFT.mat')
% Kd=NUFTData['Kd']
% P=NUFTData['P']
% SN=NUFTData['SN']
% Trajm2=NUFTData['Trajm2']
% 
% NUFTData=scipy.io.loadmat(BaseNUFTDataP + 'TrajForNUFT.mat')
% Traj=NUFTData['Trajm2'][0:2,:]

% RealData/RealDataForNN.mat
% ifilename=myParams.myDict['RealDataFN']
% RealData=scipy.io.loadmat(ifilename)
% RealData=RealData['Data']
%%
HMS=load('ForMLN.mat');
%%
BaseOutDir='HMSy';
mkdir(BaseOutDir);
system(['chmod -R 777 ' BaseOutDir]);
nCh=13;
batchSize=16;
nTS=15;
SensCC=squeeze(HMS.Sensr(:,:,1:nCh)); % [X Y Ch]
SensMsk=grmss(SensCC,3)>0.01; % [X Y]
save([BaseOutDir '/SensCC1.mat'],'SensCC','SensMsk');

WhichRep=1;
% AcqDwellTime_us=2*1.1;
AcqDwellTime_us=3*1.1;
% TrajPartToUse=1:24000;
TrajPartToUse=1:45501;
% CurSig=     HMS.sig(1,TrajPartToUse(1:2:end),WhichRep,1:nCh)+...
%             HMS.sig(1,TrajPartToUse(2:2:end),WhichRep,1:nCh);
CurSig=     HMS.sig(1,TrajPartToUse(1:3:end),WhichRep,1:nCh)+...
            HMS.sig(1,TrajPartToUse(2:3:end),WhichRep,1:nCh)+...
            HMS.sig(1,TrajPartToUse(3:3:end),WhichRep,1:nCh);
CurSig=squeeze(CurSig);
tmp=Row(CurSig);
tmp2=[real(tmp) imag(tmp)]*600;
Data=tmp2;
Data(batchSize,end)=0;
save([BaseOutDir '/RealDataForNN.mat'],'Data');
nTrajA=size(CurSig,1);
TimePoints_ms=(1:nTrajA)*AcqDwellTime_us/1000;
TimePoints_ms3=permute(TimePoints_ms,[1 3 2]);
TS_TimePoints=linspace(0,TimePoints_ms(end),nTS);
TS_TimePoints3=permute(TS_TimePoints,[1 3 2]);
TSBF=GetTSCoeffsByLinear(nTrajA,nTS).';
WhichRSToUse=1;
TSC=exp(-TS_TimePoints3./HMS.UpdatedT2SMap_ms_RS(:,:,WhichRSToUse)).*exp(-1i*2*pi*HMS.UpdatedB0Map_RS(:,:,WhichRSToUse).*TS_TimePoints3/1e3);
% TSBF: [15×7162 double]
% TSC: [128×128×15 double]
B0_Hz=HMS.UpdatedB0Map_RS(:,:,WhichRSToUse);
save([BaseOutDir '/B0TS.mat'],'TSBF','TSC','B0_Hz');
%
% Traj=(HMS.TrajM(WhichRep,TrajPartToUse(1:2:end))+HMS.TrajM(WhichRep,TrajPartToUse(2:2:end)))/2;
Traj=(HMS.TrajM(WhichRep,TrajPartToUse(1:3:end))+HMS.TrajM(WhichRep,TrajPartToUse(2:3:end))+HMS.TrajM(WhichRep,TrajPartToUse(3:3:end)))/3;
Sz128=gsize(SensCC,1:2);
clear Trajm2
Trajm2(1,:)=real(Traj);
Trajm2(2,:)=imag(Traj);
[FesNUFTOp,st] = nuFTOperator(BART2Fes_NUFT_Idxs(Trajm2,Sz128),Sz128);
Kd=st.nufftStruct.Kd;
SN=st.nufftStruct.sn;
P=st.nufftStruct.p/sqrt(prod(Sz128));
save([BaseOutDir '/TrajForNUFT.mat'],'Trajm2','SN','Kd','P');

TimePoints_ms=(1:nTrajA)*AcqDwellTime_us/1000;
TS_TimePoints=linspace(0,TimePoints_ms(end),nTS);
TSstr=strrep(num2str(TS_TimePoints,'%3.5f,'),' ','');
TSstr=['TimePoints_ms ' TSstr(1:end-1)];
disp(TSstr)
%%
% TSB3=permute(B0TS.TSB,[1 3 2]);

% MChangedB0X=squeeze(sum(MChangedB0.*TSB3,3));
% %%
% clear Data
% for i=1:12
%     tmp=Row(MChangedB0X(:,:,i))*(12/16);
%     Data(i,:)=[real(tmp) imag(tmp)];
% end
% Data(16,end)=0;
% RealDataFN=['Sli6_r04.mat'];
% save(RealDataFN,'Data');
