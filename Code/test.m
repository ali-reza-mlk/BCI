%% Normal CSP (unused)

c1 = zeros(64); c2 = zeros(64);
for i = 1:m/5
    c1 = c1+(x2(65:end-64,:,i)'*x2(65:end-64,:,i))/trace(x2(65:end-64,:,i)'*x2(65:end-64,:,i));
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

%% CSP Projection
Nx1 =[];Nx2=[];

for i=1:m/5
    Nx1 = cat(3,Nx1,x2(:,:,i)*V);
    Nx2 = cat(3,Nx2,x4(:,:,i)*V);
end

Nxv1 =[];Nxv2=[];
for i=1:150
    Nxv1 = cat(3,Nxv1,xv2(:,:,i)*V);
    Nxv2 = cat(3,Nxv2,xv4(:,:,i)*V);
end

%% PAC
K1=zeros(65,65,1,m/5);K2=zeros(65,65,1,m/5);

for u=1:m/5

c1=cwt(Nx1(65:end-64,2,u));
c2=cwt(Nx1(65:end-64,1,u));
c4=cwt(Nx2(65:end-64,2,u));
c5=cwt(Nx2(65:end-64,1,u));

for j=1:65
for i=1:65
    A = (abs(c1(j,:))-mean(abs(c1(j,:))));
    B = c2(i,:);
    K1(i,j,1,u) =  A*B'/norm(A)/norm(B);
    A = (abs(c4(j,:))-mean(abs(c4(j,:))));
    B = c5(i,:);
    K2(i,j,1,u) =  A*B'/norm(A)/norm(B);
end
end
clc
disp(u)
end

Kv1=zeros(65,65,2,150);Kv2=zeros(65,65,2,150);
for u=1:mv/5

c1=cwt(Nxv1(65:end-64,2,u));
c2=cwt(Nxv1(65:end-64,1,u));
c4=cwt(Nxv2(65:end-64,2,u));
c5=cwt(Nxv2(65:end-64,1,u));

for j=1:65
for i=1:65
    A = (abs(c1(j,:))-mean(abs(c1(j,:))));
    B = c2(i,:);
    Kv1(i,j,1,u) =  A*B'/norm(A)/norm(B);
    A = (abs(c4(j,:))-mean(abs(c4(j,:))));
    B = c5(i,:);
    Kv2(i,j,1,u) =  A*B'/norm(A)/norm(B);
end
end
clc
disp(u)

end
disp('done!')
%% TEST
q=cat(4,K1,K2);
qv=cat(4,Kv1,Kv2);


for i=1:1800
    s = q(:,:,1,i);
    
    s = abs(s);
    s = (s-mean(s,'all'))/var(s,[],'all');
    
    q(:,:,1,i) = s;
end

for i=1:300
   s = qv(:,:,1,i);
   
   s = abs(s);
   s = (s-mean(s,'all'))/var(s,[],'all');
   
   qv(:,:,1,i) = s;
end



for i=1:64
    subplot(8,8,i)
    image(q(:,:,1,i)*200)
    sgtitle('class 2')
    axis off
end
figure
for i=1:64
    subplot(8,8,i)
    image(q(:,:,1,i+900)*200)
    sgtitle('class 4')
    axis off
end
%% Check
figure
close all
for i=1:15
    qq= mean((q(:,:,60*(i-1)+1+900:60*i+900)),3);
    subplot(3,5,i)
    image(abs(qq*30))
    title(['sub' ,num2str(i)])
    
end
sgtitle('4')
figure
for i=1:15
    qq= mean((q(:,:,60*(i-1)+1:60*i)),3);
    
    subplot(3,5,i)
    image(abs(qq*30))
    title(['sub' ,num2str(i)])
    
end
sgtitle('2')
figure
for i=1:15
    qq= mean((qv(:,:,10*(i-1)+1+150:10*i+150)),3);
    
    subplot(3,5,i)
    image(abs(qq*30))
    title(['sub' ,num2str(i)])
    
end
sgtitle('v4')
figure
for i=1:15
    qq= mean((qv(:,:,10*(i-1)+1:10*i)),3);
    
    subplot(3,5,i)
    image(abs(qq*30))
    title(['sub' ,num2str(i)])
    
end
sgtitle('v2')
