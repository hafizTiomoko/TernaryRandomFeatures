%% real data: sparse, quantized or binary
close all; clear; clc
rng(1)
testcase = 'quick'; %'output'
f_choice = 'relu';
f_choice = 'ternary';
%f_choice = 'sign';
%f_choice = 'abs';
%f_choice = 'binary';
%f_choice = 'cos';


top = 50;
switch f_choice
    case 'lin'
        f = @(t) t;
    case 'sparse'
        f = @(t) t.*(1 + sign(abs(t)-sqrt(2)*s))/2;
        %ratio = 1./( 1 - erf(s) + 2*s.*exp(-s.^2)/sqrt(pi) );
    case 'sign'
        s=5;
        f = @(t) sign(t).*(1 + sign(abs(t)-sqrt(2)*s))/2;
        %f = @(t) sign(t);
        %ratio = pi*(1 - erf(s)).*exp(2*s.^2)/2;
    case 'relu'
        f = @(t) max(t,0);
    case 'cos'
        f = @(t) cos(t);
    case 'abs'
        f = @(t) abs(t);
    case 'binary'
        f = @(x, s) (x>(sqrt(2)*s)) + (x<(-sqrt(2)*s));
    case 'ternary'
        f = @(x, s_minus, s_plus, r, t) (t.*((x>(sqrt(2)*s_plus))) - r.*t.*(x<(sqrt(2)*s_minus)));
    case 'quant'
        M = 2;
        if s == 0
            f = @(t) sign(t);
        else
            f = @(x) (floor(x*2^(M-2)/sqrt(2)/s) + 1/2)/(2^(M-2)).*(abs(x)<=sqrt(2)*s) + sign(x).*(abs(x)>sqrt(2)*s);
        end
        %             switch M
        %                 case 2
        %                     a1_s = @(s) sqrt(2/pi)*(1/2+exp(-s.^2)/2);
        %                     nu_s = @(s) 1-erf(s)*3/4;
        %                 case 3
        %                     a1_s = @(s) sqrt(2/pi)*(1/4 + exp(-s.^2/4)/2 + exp(-s.^2)/4);
        %                     nu_s = @(s) 1 - erf(s)*7/16 - erf(s/2)/2;
        %             end
        %             ratio = nu_s(s)/(a1_s(s)^2);
end
tic;
k = 2;
cs = 1/k*ones(1,k);%[1/2 1/2];

N = 256;
switch testcase
    case 'quick'
        n = 1000;
        nb_average_loop  = 10;
    case 'output'
        n = 2048;
        nb_average_loop  = 100;
end
ns = int32(cs*n);
%
k = length(cs); % nb of classes
v = [-ones(n/2,1);ones(n/2,1)];

data_choice = 'EEG'; % 'MNIST', 'fashion', 'kannada', 'EEG'
switch data_choice
    case 'MNIST'
        selected_labels=[1 7]; % mean [0 1], [5 6]
        init_images = loadMNISTImages('./datasets/MNIST/train-images-idx3-ubyte');
        init_labels = loadMNISTLabels('./datasets/MNIST/train-labels-idx1-ubyte');
        [labels,idx_init_labels]=sort(init_labels,'ascend');
        images=init_images(:,idx_init_labels);
        init_n=length(images(1,:));
                
        p=length(images(:,1));
                 
        noise_level_dB=-10;
        noise_level=10^(noise_level_dB/10);
        Noise = rand(p,init_n)*sqrt(12)*sqrt(noise_level/var(images(:)));
        
        %%% Add noise to images
        %images=images+Noise;

        mean_images=mean(images,2);
        norm_images=0;
        for i=1:init_n
            norm_images=norm_images+1/init_n*norm(images(:,i)-mean_images);
        end
        images=images/norm_images*sqrt(p);
        
        j=1;
        for i=selected_labels
            MNIST{j}=images(:,labels==i);
            j=j+1;
        end
    case 'fashion'
        selected_labels=[1 7];
        init_images = loadMNISTImages('./datasets/fashion-MNIST/train-images-idx3-ubyte');
        init_labels = loadMNISTLabels('./datasets/fashion-MNIST/train-labels-idx1-ubyte'); 
        [labels,idx_init_labels]=sort(init_labels,'ascend');
        images=init_images(:,idx_init_labels);
        init_n=length(images(1,:));
                
        p=length(images(:,1));
                 
        noise_level_dB=-10;
        noise_level=10^(noise_level_dB/10);
        Noise = rand(p,init_n)*sqrt(12)*sqrt(noise_level/var(images(:)));
        
        %%% Add noise to images
        %images=images+Noise;

        mean_images=mean(images,2);
        norm_images=0;
        for i=1:init_n
            norm_images=norm_images+1/init_n*norm(images(:,i)-mean_images);
        end
        images=images/norm_images*sqrt(p);
        
        j=1;
        for i=selected_labels
            fashion{j}=images(:,labels==i);
            j=j+1;
        end
    case 'Kuzushiji'
        selected_labels=[3 4];
        noise_level_dB= -Inf; %-Inf, -3 or 3
        init_data = loadMNISTImages('../datasets/Kuzushiji-MNIST/train-images-idx3-ubyte');
        init_labels = loadMNISTLabels('../datasets/Kuzushiji-MNIST/train-labels-idx1-ubyte');
    case 'kannada'
        selected_labels=[4 8];
        noise_level_dB= -15; %-Inf, -3 or 3
        init_data = loadMNISTImages('../datasets/kannada-MNIST/train-images-idx3-ubyte');
        init_labels = loadMNISTLabels('../datasets/kannada-MNIST/train-labels-idx1-ubyte');
    case 'CIFAR'
        selected_labels=[0 1];
        noise_level_dB= -Inf; %-Inf, -3 or 3
        data1 = readNPY('../datasets/CIFAR_feature/Real_hamburger_googlenet.npy');
        data2 = readNPY('../datasets/CIFAR_feature/Real_coffee_googlenet.npy');
%         data1 = readNPY('../datasets/CIFAR_feature/Real_pizza_googlenet.npy');
%         data2 = readNPY('../datasets/CIFAR_feature/Real_daisy_googlenet.npy');

    case 'EEG'
        %init_images=processing_EEG();
        load('./datasets/EEG.mat')
        init_images=data;
        init_labels = [zeros(4097,1);ones(4097,1)];
        [labels,idx_init_labels]=sort(init_labels,'ascend');
        init_labels = labels;
        init_data = init_images(:,idx_init_labels);
        selected_labels=[0 1];

        
        noise_level_dB=-10;
        [labels,idx_init_labels]=sort(init_labels,'ascend');
        images=init_images(:,idx_init_labels);
        init_n=length(images(1,:));
                
        p=length(images(:,1));
        noise_level=10^(noise_level_dB/10);
        Noise = rand(p,init_n)*sqrt(12)*sqrt(noise_level/var(images(:)));
        
        %%% Add noise to images
        %images=images+Noise;

        mean_images=mean(images,2);
        norm_images=0;
        for i=1:init_n
            norm_images=norm_images+1/init_n*norm(images(:,i)-mean_images);
        end
        images=images/norm_images*sqrt(p);
        
        j=1;
        for i=selected_labels
            EEG{j}=images(:,labels==i);
            j=j+1;
        end
end

%[labels,idx_init_labels]=sort(init_labels,'ascend');
%data=init_data(:,idx_init_labels);

%init_n=length(data(1,:));
%p=length(data(:,1));

if length(selected_labels) ~= k
    error('Error: selected labels and nb of classes not equal!')
end

% Data preprecessing
% data = data/max(data(:));
% mean_data=mean(data,2);
% norm2_data=0;
% for i=1:init_n
%     norm2_data=norm2_data+1/init_n*norm(data(:,i)-mean_data)^2;
% end
% data=(data-mean_data*ones(1,size(data,2)))/sqrt(norm2_data)*sqrt(p);
% 
% 
% selected_data = cell(k,1);
% cascade_selected_data=[];
% j=1;
% for i=selected_labels
%     selected_data{j} = data(:,labels==i);
%     j = j+1;
% end


X=zeros(p,n);
n_train = n/2;
n_test = n/2;
X1_train=zeros(p,cs(1)*n_train);
X2_train=zeros(p,cs(2)*n_train);
X1_test=zeros(p,cs(1)*n_test);
X2_test=zeros(p,cs(2)*n_test);
switch data_choice
    case 'MNIST'
        for i=1:k
            X(:,sum(cs(1:(i-1)))*n+1:sum(cs(1:i))*n)=MNIST{i}(:,1:n*cs(i));
        end
        i = 1;
        data = MNIST{i}(:,randperm(size(MNIST{i},2)));
        X1_train(:,sum(cs(1:(i-1)))*n_train+1:sum(cs(1:i))*n_train)=data(:,1:n_train*cs(i)); 
        X1_test(:,sum(cs(1:(i-1)))*n_test+1:sum(cs(1:i))*n_test)=data(:,n_train*cs(i)+1:n/2); 
        i=2;
        data = MNIST{i}(:,randperm(size(MNIST{i},2)));
        X2_train(:,sum(cs(1:(i-1)))*n_train+1:sum(cs(1:i))*n_train)=data(:,1:n_train*cs(i)); 
        X2_test(:,sum(cs(1:(i-1)))*n_test+1:sum(cs(1:i))*n_test)=data(:,n_train*cs(i)+1:n/2);
        X_test = [X1_test X2_test];
        Y_test = [selected_labels(1)*ones(size(X1_test,2),1);selected_labels(2)*ones(size(X2_test,2),1)];
    case 'fashion'
        for i=1:k
            X(:,sum(cs(1:(i-1)))*n+1:sum(cs(1:i))*n)=fashion{i}(:,1:n*cs(i));
        end
        i = 1;
        data = fashion{i}(:,randperm(size(fashion{i},2)));
        X1_train(:,sum(cs(1:(i-1)))*n_train+1:sum(cs(1:i))*n_train)=data(:,1:n_train*cs(i)); 
        X1_test(:,sum(cs(1:(i-1)))*n_test+1:sum(cs(1:i))*n_test)=data(:,n_train*cs(i)+1:n/2); 
        i=2;
        data = fashion{i}(:,randperm(size(fashion{i},2)));
        X2_train(:,sum(cs(1:(i-1)))*n_train+1:sum(cs(1:i))*n_train)=data(:,1:n_train*cs(i)); 
        X2_test(:,sum(cs(1:(i-1)))*n_test+1:sum(cs(1:i))*n_test)=data(:,n_train*cs(i)+1:n/2);
        X_test = [X1_test X2_test];
        Y_test = [selected_labels(1)*ones(size(X1_test,2),1);selected_labels(2)*ones(size(X2_test,2),1)];
    case 'EEG'
        for i=1:k
            X(:,sum(cs(1:(i-1)))*n+1:sum(cs(1:i))*n)=EEG{i}(:,1:n*cs(i));
        end
        i = 1;
        data = EEG{i}(:,randperm(size(EEG{i},2)));
        X1_train(:,sum(cs(1:(i-1)))*n_train+1:sum(cs(1:i))*n_train)=data(:,1:n_train*cs(i)); 
        X1_test(:,sum(cs(1:(i-1)))*n_test+1:sum(cs(1:i))*n_test)=data(:,n_train*cs(i)+1:n/2); 
        i=2;
        data = EEG{i}(:,randperm(size(EEG{i},2)));
        X2_train(:,sum(cs(1:(i-1)))*n_train+1:sum(cs(1:i))*n_train)=data(:,1:n_train*cs(i)); 
        X2_test(:,sum(cs(1:(i-1)))*n_test+1:sum(cs(1:i))*n_test)=data(:,n_train*cs(i)+1:n/2);
        X_test = [X1_test X2_test];
        Y_test = [selected_labels(1)*ones(size(X1_test,2),1);selected_labels(2)*ones(size(X2_test,2),1)];
end


%mean(diag(X*X'))/n
tau_est = trace(X*X'/n)/p
%%%%%%%%%%% PERFS %%%%%%%%%%% 
%     D = diag((diag(X'*X)).^(-1/2));
%     X = X*D*sqrt(p);
%     X = (mapstd(X'))';
% Add Gaussian noise to data
noise_level=10^(noise_level_dB/10);
%Noise = rand(p,n)*sqrt(12)*sqrt(noise_level*var(X(:)));
%X = X*(eye(n) - ones(n)/n);
%X=X+Noise;
%X = randn(p,n)+means*(v');

%K = f(X'*X/sqrt(p))/sqrt(p);
% fashion : -0.1710    0.7618    1.3604
% mnist: -0.22 0.815 0.41 1.46
% eeg: -0.8041    2.1842    3.7763

%t = 1.47;
%s = 4.60;
switch data_choice
    case 'MNIST'
        s_minus = 0.0090;
        s_plus = 0.8402;
        r = 0.2853;
        t = 1.4253;
    case 'fashion'
        s_minus = -0.0205;
        s_plus = 0.9621;
        r = 0.3025;
        t = 1.7524;
    case 'EEG'
        s_minus = -0.0091;
        s_plus = 0.9166;
        r = 0.2963;
        t = 1.6767;
end
%s_minus = 0;
%s_plus = 0;
%r = 1;
%t = 1;


rte=0;

for data_index = 1:nb_average_loop
    %         D = diag((diag(X'*X)).^(-1/2));
    %         X = X*D*sqrt(p);
    %         X = (mapstd(X'))';
    noise_level=10^(noise_level_dB/10);
    Noise = rand(p,n)*sqrt(12)*sqrt(noise_level*var(X(:)));
    %X = X*(eye(n) - ones(n)/n);
    %X=X+Noise;
    %X = randn(p,n)+means*(v');

    %K = f(X'*X/sqrt(p))/sqrt(p);
W = sign(randn(N,p));
%W = randn(N,p);
%b1 = 2*pi*rand(N,size(X1_train,2))-pi; %b1 = rand(N,size(X1_train,2));
%b2= 2*pi*rand(N,size(X2_train,2))-pi;%b2 = rand(N,size(X2_train,2));

switch f_choice
    case 'sign'
        K11 = f(W*X1_train)'*f(W*X1_train)/p;
		K22 = f(W*X2_train)'*f(W*X2_train)/p;
		K12 = f(W*X1_train)'*f(W*X2_train)/p;
        XX = f(W*X1_train);
        YY = f(W*X2_train);
    case 'cos'
        K11 = f(W*X1_train)'*f(W*X1_train)/p;
		K22 = f(W*X2_train)'*f(W*X2_train)/p;
		K12 = f(W*X1_train)'*f(W*X2_train)/p;
        XX = f(W*X1_train);
        YY = f(W*X2_train);
    case 'relu' 
        K11 = f(W*X1_train)'*f(W*X1_train)/p;
		K22 = f(W*X2_train)'*f(W*X2_train)/p;
		K12 = f(W*X1_train)'*f(W*X2_train)/p;
        XX = f(W*X1_train);
        YY = f(W*X2_train);
    case 'ternary'
        K11 = f(W*X1_train, s_minus, s_plus, r, t)'*f(W*X1_train, s_minus, s_plus, r, t)/p;
		K22 = f(W*X2_train, s_minus, s_plus, r, t)'*f(W*X2_train, s_minus, s_plus, r, t)/p;
		K12 = f(W*X1_train, s_minus, s_plus, r, t)'*f(W*X2_train, s_minus, s_plus, r, t)/p;
        XX = f(W*X1_train, s_minus, s_plus, r, t);
        YY = f(W*X2_train, s_minus, s_plus, r, t);
    case 'abs'
        K11 = f(W*X1_train)'*f(W*X1_train)/p;
		K22 = f(W*X2_train)'*f(W*X2_train)/p;
		K12 = f(W*X1_train)'*f(W*X2_train)/p;
        XX = f(W*X1_train);
        YY = f(W*X2_train);
    case 'binary'
        K11 = f(W*X1_train+b1)'*f(W*X1_train+b1)/p;
		K22 = f(W*X2_train+b2)'*f(W*X2_train+b2)/p;
		K12 = f(W*X1_train+b1)'*f(W*X2_train+b2)/p;
end

[a, b] = rcca_fit(K11, K22, K12,top);
%[a, b] = rcca_fit(XX,YY);
[x_cor, y_cor] = rcca_eval(a, b, f, X1_test,X2_test, W, f_choice, s_minus, s_plus, r, t);
functions = @(i) abs(correlation(x_cor(:,i), y_cor(:,i)));
tab = 1:1:top;
r_te = rte + sum(arrayfun(functions,tab))/nb_average_loop;
end
r_te


function [L, M] = rcca_fit(Kx,Ky,Kxy,top)
  [L, M]  = geigen(Kxy, Kx, Ky, top); % rcc(augx(x),augy(y),1e-10,1e-10,top);
  %cor=sum(abs(values(1:top)));
end

% function [L, M] = rcca_fit(X,Y)
%   [L,M] = canoncorr(X,Y);
% end

function [xx, yy] = rcca_eval(a, b, f, x, y, W, f_choice, s_minus, s_plus, r, t)
switch f_choice
case 'ternary'
        xx = f(W*x, s_minus, s_plus, r, t)*a;
        yy = f(W*y, s_minus, s_plus, r, t)*b;
    case 'cos'
        xx = f(W*x)*a;
        yy = f(W*y)*b;
        
    
    case 'relu'
        xx = f(W*x)*a;
        yy = f(W*y)*b;
        
    case 'sign'
        xx = f(W*x)*a;
        yy = f(W*y)*b;
        
    case 'abs'
        xx = f(W*x)*a;
        yy = f(W*y)*b;
        
end

	
end

function [L, M]=geigen(A, B, C, top)
	p = size(B,1);
	q = size(C,1);
% 	s = min(p,q);
% 	B = (B+B')/2;
% 	C = (C+C')/2;
% 	B = nearestSPD(B);
% 	Bfac = chol(B);
% 	C = nearestSPD(C);
% 	Cfac = chol(C);
% 	Bfacinv = inv(Bfac);
% 	Cfacinv = inv(Cfac);
% 	D = (Bfacinv')*A*Cfacinv;
% 	
% 	if p>=q
% 		[U,S,V] = svds(D, top);
% 		values = diag(S);
% 		L = Bfacinv*U;
% 		M = Cfacinv*V;
% 	else
% 		[U,S,V] = svds(D', top);
% 		values = diag(S);
% 		L = Bfacinv*V;
% 		M = Cfacinv*U;
%     end
    
    gamma = 1e-6;
    AA = [zeros(size(A,1),size(A,1)),A;A',zeros(size(A,2),size(A,2))];
    BB = [B+gamma*eye(size(B,1),size(B,1)),zeros(size(B,1),size(C,1));zeros(size(C,1),size(B,1)),C+gamma*eye(size(C))];
    [V,D,W] = eig(AA,BB);
    L = W(1:p,:);
    M = W(p+1:end,:);
end

function [cor] = correlation(X,Y)
	covariance = bsxfun(@minus,X,mean(X))'*bsxfun(@minus,Y,mean(Y))/(size(X,1)-1);
	cor = covariance/(sqrt(var(X)*var(Y)));
end
