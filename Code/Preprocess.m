%% Extract data
clc
close all
clear
[~,path] = uigetfile;

x=[]; trial=[];


for i=1:9
Data = load([path,'\Data_Sample0',num2str(i)]);
x = cat(3,x,Data.epo_train.x);
trial = cat(2,trial,Data.epo_train.y);
end
for i=10:15
Data = load([path,'\Data_Sample',num2str(i)]);
x = cat(3,x,Data.epo_train.x);
trial = cat(2,trial,Data.epo_train.y);
end
t = Data.epo_train.t;
%% Vlidation data
[~,path] = uigetfile;
xvalid = []; tvalid=[];
for i=1:9
Data = load([path,'\Data_Sample0',num2str(i)]);
xvalid = cat(3,xvalid,Data.epo_validation.x);
tvalid = cat(2,tvalid,Data.epo_validation.y);
end
for i=10:15
Data = load([path,'\Data_Sample',num2str(i)]);
xvalid = cat(3,xvalid,Data.epo_validation.x);
tvalid = cat(2,tvalid,Data.epo_validation.y);
end

%% Filter
load('LPF45'); load('HPF1');

figure
subplot(2,1,1)
plot(t,x(:,10,2))
subplot(2,1,2)
periodogram(x(:,10,2),[],795,256)

%%%%%% Perprocess
[l,n,m]=size(x);
X = zeros(ceil(l/2),n,m);
for i=1:64
    for j=1:m
        filX = conv(LPF45,x(:,i,j)');
        filX = filX(109:end-108);
        filX = downsample(filX,2);
        filX = conv(HPF1,filX);
        filX = filX(123:end-122);
        X(:,i,j) = filX;
    end
end

[lv,nv,mv]=size(xvalid);
Xv = zeros(ceil(lv/2),nv,mv);
for i=1:64
    for j=1:mv
        filX = conv(LPF45,xvalid(:,i,j)');
        filX = filX(109:end-108);
        filX = downsample(filX,2);
        filX = conv(HPF1,filX);
        filX = filX(123:end-122);
        Xv(:,i,j) = filX;
    end
end

figure
subplot(2,1,1)
plot(downsample(t,2),X(:,10,2))
subplot(2,1,2)
periodogram(X(:,10,2),[],400,128)

%% Separate
clc

xv1=[]; xv2=[]; xv3=[]; xv4=[];xv5=[];
for i=1:150
    if tvalid(1,i)
        xv1 = cat(3,xv1,Xv(:,:,i));
    end
    if tvalid(2,i)
        xv2 = cat(3,xv2,Xv(:,:,i));
    end
    if tvalid(3,i)
        xv3 = cat(3,xv3,Xv(:,:,i));
    end
    if tvalid(4,i)
        xv4 = cat(3,xv4,Xv(:,:,i));
    end
    if tvalid(5,i)
        xv5 = cat(3,xv5,Xv(:,:,i));
    end
end
x1=[]; x2=[]; x3=[]; x4=[];x5=[];
for i=1:m
    if tvalid(1,i)
        x1 = cat(3,x1,X(:,:,i));
    end
    if tvalid(2,i)
        x2 = cat(3,x2,X(:,:,i));
    end
    if tvalid(3,i)
        x3 = cat(3,xv3,X(:,:,i));
    end
    if tvalid(4,i)
        x4 = cat(3,xv4,X(:,:,i));
    end
    if tvalid(5,i)
        x5 = cat(3,xv5,X(:,:,i));
    end
end
%%
S1 = zeros(64);
S2 = zeros(64);
for i=1:m/5
    S1 = S1+x1(:,:,i)'*x1(:,:,i);
    S2 = S2+x2(:,:,i)'*x2(:,:,i);
end
S1 = S1/i;
S2 = S2/i;

P = (S1+S2)^-.5;
Sp1 = P*S1*P';
Sp2 = P*S2*P';

[~,Si1,Si2] = D2epo(x1,x2); %% 2 epochs for deflation_divCSP

%%
d=24; % number of outputs' channels
Vk = KLsub_divCSP(P,Sp1,Sp2,Si1,Si2,d); 

%V = deflation_divCSP(x1,x2,Sp1,Sp2,10);
%% Functions

function V = KLsub_divCSP(P,Sp1,Sp2,Si1,Si2,d)
    
    D = length(Sp1);
    l = .5;
    t=.001; % Change untile convergence
    Id = eye(d,D);
    R = eye(D);
    
    e=[0.1,0];
    while e(1)>1e-8 && e(1)<1
        
        L = KLdivCSP_ws(Sp1,Sp2,Si1,Si2,R,d,l);
        for m=1:D
            for n=1:m
              L(m,n) = -L(m,n);
            end
        end
        
        U=eye(D)-L*t;
        R = U*R;
        o= Sp1 - U*Sp1*U';
        e(2)=e(1);
        e(1)=0;
        for m=1:d
            for n=1:d
                e(1)=e(1)+o(m,n)^2;
            end
        end
        Sp1 = U*Sp1*U';
        Sp2 = U*Sp2*U';
        disp(e(1))
    end
    V=Id*R*P;
    V=V';
    [G,~] = eig(V'*Sp1*V);
    
    V = V*G;
    
end





function C = KLCSPterm(Sp1,Sp2,R,d,l)
    D = length(Sp1);
    Id = eye(d,D);
    Sb1 = Id*R*Sp1*R'*Id';
    Sb2 = Id*R*Sp2*R'*Id';
        
    a = Sb2^-1*Id*Sp2 - Sb1^-1*Sb2*Sb1^-1*Id*Sp1 ...
        + Sb1^-1*Id*Sp1 - Sb2^-1*Sb1*Sb2^-1*Id*Sp2;
    
    C = (1-l)*Id'*a*R;

end


function L = KLdivCSP_ws(Sp1,Sp2,Si1,Si2,R,d,l)
    D = length(Sp1);
    Id = eye(d,D);
    C = KLCSPterm(Sp1,Sp2,R,d,l);
    
    Sb1 = Id*R*Sp1*R'*Id';
    Sb2 = Id*R*Sp2*R'*Id';
    a=zeros(D);
    for i=1:2
        S1 = Id*R*Si1(:,:,i)*R'*Id';
        S2 = Id*R*Si2(:,:,i)*R'*Id';
        a1 = Id'*(Sb1^-1*Id*Sp1 - Sb1^-1*S1*Sb1^-1*Id*Sp1)*R;
        a2 = Id'*(Sb2^-1*Id*Sp2 - Sb2^-1*S2*Sb2^-1*Id*Sp2)*R;
        a = a+a1+a2;
    end
    L = C - l/2/5*a;
end



function V = deflation_divCSP(x1,x2,Sp1,Sp2,d)
    [~,~,m] = size(x1);
    B = eye(64);
    v = zeros(64,d);
    X1 = x1; X2 = x2;
    for i=1:d
        [P,Si1,Si2] = D2epo(X1,X2);
        if i==1
            p=P;
        end
        w = KLsub_divCSP(P,Sp1,Sp2,Si1,Si2,1);
        disp(size(w))
        W = null(w');
        Sp1 = W'*Sp1*W;
        Sp2 = W'*Sp2*W;
        v(:,i) = B*w;
        B = B*W;
        XX1=[]; XX2=[];
        for j=1:m
            XX1 = cat(3,XX1,X1(:,:,j)*W);
            XX2 = cat(3,XX1,X2(:,:,j)*W);
        end
        X1 = XX1;
        X2 = XX2;
        disp(i)
    end
    V = p*v;
end




function [P,Si1,Si2] = D2epo(x1,x2)
    
    [l,n,m]=size(x1);
    Si1 = zeros(n,n,2);
    Si2 = zeros(n,n,2);
    
    S1 = zeros(n);
    S2 = zeros(n);
    for i=1:m
        S1 = S1+x1(:,:,i)'*x1(:,:,i);
        S2 = S2+x2(:,:,i)'*x2(:,:,i);
    end
    S1 = S1/i;
    S2 = S2/i;

    P = (S1+S2)^-.5;
    for i=1:m
        for j=1:2

            idx = 1:floor(l/2);
            index = idx+(j-1)*floor(l/2);
            Si1(:,:,j) = Si1(:,:,j)+x1(index,:,i)'*x1(index,:,i);
            Si2(:,:,j) = Si2(:,:,j)+x2(index,:,i)'*x2(index,:,i);
        end
    end
    for j=1:2
        p = (Si1(:,:,j)+Si2(:,:,j))^-.5;
        Si1(:,:,j) = p*Si1(:,:,j)*p';
        Si2(:,:,j) = p*Si2(:,:,j)*p';
    end
end