%% Average of coherence for all pairs of channels
for i=2:64
    for j=1:i-1
        a1=zeros(128,1)';a2=zeros(128,1)';
        for u=1:m/5
a1 = a1+coherence_MVDR(nx1(:,i,u),nx1(:,j,u),64,100);
a2 = a2+coherence_MVDR(nx2(:,i,u),nx2(:,j,u),64,100);

        end
plot(a1(1:64)/u)
hold on
plot(a2(1:64)/u)
title([i,j])
axis([0,64,0,1])
hold off
pause(.1)
if max(abs(a1-a2))>.07*u
    disp([i,j])
end
end

end

function [MSC]=coherence_MVDR(x1,x2,L,K)


% L is the length of MVDR filter or window length
% K is the resolution (the higher K, the better resolution)

%initialization
xx1     = zeros(L,1);
xx2     = zeros(L,1);
r11     = zeros(L,1);
r22     = zeros(L,1);
r12     = zeros(L,1);
r21     = zeros(L,1);

%construction of the Fourier Matrix
F       = zeros(L,K);
l       = [0:L-1]';
f       = exp(2*pi*l*j/K);
for k = 0:K-1
    F(:,k+1) = f.^k;
end
F       = F/sqrt(L);

%number of samples, equal to the lenght of x1 and x2
n       = length(x1);

for i = 1:n
    xx1 = [x1(i);xx1(1:L-1)];
    xx2 = [x2(i);xx2(1:L-1)];
    r11 = r11 + xx1*conj(xx1(1));
    r22 = r22 + xx2*conj(xx2(1));
    r12 = r12 + xx1*conj(xx2(1));
    r21 = r21 + xx2*conj(xx1(1));
end
r11 = r11/n;
r22 = r22/n;
r12 = r12/n;
r21 = r21/n;
%
R11 = toeplitz(r11);
R22 = toeplitz(r22);
R12 = toeplitz(r12,conj(r21));
%
%for regularization
Dt1     = 0.01*r11(1)*diag(diag(ones(L)));
Dt2     = 0.01*r22(1)*diag(diag(ones(L)));
%
Ri11    = inv(R11 + Dt1);
Ri22    = inv(R22 + Dt2);
Rn12    = Ri11*R12*Ri22;
%
Si11    = real(diag(F'*Ri11*F));
Si22    = real(diag(F'*Ri22*F));
S12     = diag(F'*Rn12*F);
%
%Magnitude squared coherence function
MSC     = real(S12.*conj(S12))./(Si11.*Si22);
end