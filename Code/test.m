clc

c1 = zeros(64); c2 = zeros(64);
for i=1:m/5
    c1 = c1+(x1(65:end-64,:,i)'*x1(65:end-64,:,i))/trace(x1(65:end-64,:,i)'*x1(65:end-64,:,i));
    c2 = c2+(x4(65:end-64,:,i)'*x4(65:end-64,:,i))/trace(x4(65:end-64,:,i)'*x4(65:end-64,:,i));
end
c1 = c1/i; c2 = c2/i;

c = c1+c2;
w = c^.5;

[v,h] = eig(w);
w = v*sqrt(h)*v^-1;

S1 = w*c1*w';
[v,h] = eig(S1);
P = v'*w;

%%
nx1 =[];nx2=[];

for i=1:m/5
    nx1 = cat(3,nx1,x1(65:end-64,:,i)*P);
    nx2 = cat(3,nx2,x4(65:end-64,:,i)*P);
end

nxv1 =[];nxv2=[];
for i=1:150
    nxv1 = cat(3,nxv1,xv1(65:end-64,:,i)*P);
    nxv2 = cat(3,nxv2,xv4(65:end-64,:,i)*P);
end



%%
K1=zeros(52,52,m/5);K2=zeros(52,52,m/5);
Kv1=zeros(52,52,150);Kv2=zeros(52,52,150);
for u=1:m/5

c1=cwt(nx1(:,21,u));
c2=cwt(nx1(:,30,u));
c3=cwt(nx2(:,21,u));
c4=cwt(nx2(:,30,u));
for j=1:52
for i=1:52
    K1(i,j,u) =  abs(c1(j,:))*c2(i,:)'/norm(abs(c1(j,:)))/norm(c2(i,:));
    K2(i,j,u) =  abs(c3(j,:))*c4(i,:)'/norm(abs(c3(j,:)))/norm(c4(i,:));
end
end
clc
u
end

%%
for u=1:150

c1=cwt(nxv1(:,21,u));
c2=cwt(nxv1(:,30,u));
c3=cwt(nxv2(:,21,u));
c4=cwt(nxv2(:,30,u));
for j=1:52
for i=1:52
    Kv1(i,j,u) =  abs(c1(j,:))*c2(i,:)'/norm(abs(c1(j,:)))/norm(c2(i,:));
    Kv2(i,j,u) =  abs(c3(j,:))*c4(i,:)'/norm(abs(c3(j,:)))/norm(c4(i,:));
end
end
clc
u
end