%% real data: sparse, quantized or binary
close all; clear; clc
rng(1)
testcase = 'quick'; %'output'
%f_choice = 'relu';
%f_choice = 'ternary';
%f_choice = 'sign';
f_choice = 'abs';
%f_choice = 'binary';

tic;
k = 4;
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

data_choice = 'fashion'; % 'MNIST', 'fashion', 'kannada', 'EEG'
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
        selected_labels=[0 1 2 7];
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
%data = data/max(data(:));
%mean_data=mean(data,2);
%norm2_data=0;
%for i=1:init_n
%    norm2_data=norm2_data+1/init_n*norm(data(:,i)-mean_data)^2;
%end
%data=(data-mean_data*ones(1,size(data,2)))/sqrt(norm2_data)*sqrt(p);

% [data,~,~] = zscore(data);
% 
% selected_data = cell(k,1);
% cascade_selected_data=[];
% j=1;
% for i=selected_labels
%     %     tmp_data=data(:,labels==i);
%     %     norm_selected_data = mean(diag(tmp_data'*tmp_data));
%     %     selected_data{j} = sqrt(p)*tmp_data/norm_selected_data;
%     
%     selected_data{j} = data(:,labels==i);
%     %cascade_selected_data = [cascade_selected_data, selected_data{j}];
%     j = j+1;
% end

X=zeros(p,n);
switch data_choice
    case 'MNIST'
        for i=1:k
            X(:,sum(cs(1:(i-1)))*n+1:sum(cs(1:i))*n)=MNIST{i}(:,1:n*cs(i));
        end
    case 'fashion'
        for i=1:k
            X(:,sum(cs(1:(i-1)))*n+1:sum(cs(1:i))*n)=fashion{i}(:,1:n*cs(i));
        end
    case 'EEG'
        for i=1:k
            X(:,sum(cs(1:(i-1)))*n+1:sum(cs(1:i))*n)=EEG{i}(:,1:n*cs(i));
        end
end

%%%%%%%%%%% PERFS %%%%%%%%%%%
store_perf = zeros(nb_average_loop,1);
store_time = zeros(nb_average_loop,2);
store_output = zeros(1,3);

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
        case 'abs'
            f = @(t) abs(t);
        case 'binary'
            f = @(x, s) (x>(sqrt(2)*s)) + (x<(-sqrt(2)*s));
        case 'ternary'
            f = @(x, s_minus, s_plus, r, t) (t*((x>(sqrt(2)*s_plus))) - r*t*(x<(sqrt(2)*s_minus)));
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

tau_est = trace(X'*X)/p/n
tau_est2 = mean(diag(X*X'))/n
%     D = diag((diag(X'*X)).^(-1/2));
%     X = X*D*sqrt(p);
%     X = (mapstd(X'))';
% Add Gaussian noise to data
noise_level=10^(noise_level_dB/10);
Noise = rand(p,n)*sqrt(12)*sqrt(noise_level*var(X(:)));
X = X*(eye(n) - ones(n)/n);
X=X+Noise;
%X = randn(p,n)+means*(v');

%K = f(X'*X/sqrt(p))/sqrt(p);
% fashion : -0.1710    0.7618    1.3604
% mnist: -0.22 0.815 0.41 1.46
% eeg: -0.8041    2.1842    3.7763
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


    W = sign(randn(N,p));
    %W = randn(N,p);
switch f_choice
    case 'sign'
        K = f(W*X)'*f(W*X)/p;
    case 'relu' 
        K = f(W*X)'*f(W*X)/p;
    case 'ternary'
        K = f(W*X, s_minus, s_plus, r, t)'*f(W*X, s_minus, s_plus, r, t)/p;
    case 'abs'
        K = f(W*X)'*f(W*X)/p;
    case 'binary'
        K = f(W*X, s)'*f(W*X, s)/p;
end
%K = K - diag(diag(K));
switch f_choice
    case 'sparse'
        prop_zero = sum(sum(K==0))/n^2;
        eps_unif = 1-prop_zero;
    case 'sign'
        prop_zero = sum(sum(K==0))/n^2;
        %eps_unif = (1-prop_zero)/64;
        store_output(3) = (1-prop_zero)/64;
    case 'relu'
        prop_zero = sum(sum(f(W*X)==0))/n^2
        store_output(3) = (1-prop_zero);
    case 'ternary'
        prop_zero = sum(sum(f(W*X, s_minus, s_plus, r, t)==0))/n^2
        store_output(3) = (1-prop_zero)/64;
    case 'abs'
        prop_zero = sum(sum(K==0))/n^2;
        store_output(3) = (1-prop_zero)/64;
    case 'binary'
        prop_zero = sum(sum(K==0))/n^2;
        store_output(3) = (1-prop_zero)/64;

end
%store_output(1,5) = 1-prop_zero;


for data_index = 1:nb_average_loop
        X=zeros(p,n);
switch data_choice
    case 'MNIST'
        for i=1:k
            X(:,sum(cs(1:(i-1)))*n+1:sum(cs(1:i))*n)=MNIST{i}(:,1:n*cs(i));
        end
    case 'fashion'
        for i=1:k
            X(:,sum(cs(1:(i-1)))*n+1:sum(cs(1:i))*n)=fashion{i}(:,1:n*cs(i));
        end
    case 'EEG'
        for i=1:k
            X(:,sum(cs(1:(i-1)))*n+1:sum(cs(1:i))*n)=EEG{i}(:,1:n*cs(i));
        end
end
    %         D = diag((diag(X'*X)).^(-1/2));
    %         X = X*D*sqrt(p);
    %         X = (mapstd(X'))';
    noise_level=10^(noise_level_dB/10);
    Noise = rand(p,n)*sqrt(12)*sqrt(noise_level*var(X(:)));
    X = X*(eye(n) - ones(n)/n);
    X=X+Noise;
    %X = randn(p,n)+means*(v');

    %K = f(X'*X/sqrt(p))/sqrt(p);
switch f_choice
    case 'sign'
        K = f(W*X)'*f(W*X)/p;
    case 'relu' 
        K = f(W*X)'*f(W*X)/p;
    case 'ternary'
        K = f(W*X, s_minus, s_plus, r, t)'*f(W*X, s_minus, s_plus, r, t)/p;
    case 'binary'
        K = f(W*X, s)'*f(W*X, s)/p;
end
    %K = sparse(K - diag(diag(K)));
    %tic
    %[hat_v,~] = eigs(K,1,'largestreal');
    %store_time(data_index,1) = toc;
    %if v'*hat_v <=0
    %    hat_v = -hat_v;
    %end
%%% k-means performance (starting from oracle solution) with kmeans_nb_vec eigenvectors
kmeans_nb_vec=k;
[U_nL,D_nL]=eigs(K,kmeans_nb_vec);
[tmpD,indD_nL]=sort(real(diag(D_nL)),'descend');
    
U_nL=U_nL(:,indD_nL);

mU_nL=zeros(k,kmeans_nb_vec);
for i=1:k
    mU_nL(i,:)=mean(U_nL(sum(cs(1:(i-1)))*n+1:sum(cs(1:i))*n,1:kmeans_nb_vec));
end
kmeans_output = kmeans(U_nL(:,1:kmeans_nb_vec),k,'Start',mU_nL);

tmp=0;
for perm=perms(1:k)'
    vec=zeros(n,1);
    for i=1:k
        vec(sum(ns(1:(i-1)))+1:sum(ns(1:i)))=perm(i)*ones(ns(i),1);
    end
    if kmeans_output'*vec>tmp
        tmp=kmeans_output'*vec;
        best_vec=vec;
    end
end

kmeans_perf = sum(best_vec==kmeans_output)/n;

%%%%%%%%%%%%%%%%%
    %Z = double(rand(n)<eps_unif);
    %B = triu(Z) + triu(Z)'-diag(diag(triu(Z)));
    %KB = (X'*X/p).*B;
    %KB = sparse(KB - diag(diag(KB)));
    %tic
    %[hat_v_KB,~] = eigs(KB,1,'largestreal');
    %store_time(data_index,2) = toc;
    %if v'*hat_v_KB <=0
    %    hat_v_KB = -hat_v_KB;
    %end
    %[U_nL,D_nL]=eigs(KB,kmeans_nb_vec);
%[tmpD,indD_nL]=sort(real(diag(D_nL)),'descend');
    
%U_nL=U_nL(:,indD_nL);
%kmeans_nb_vec=k;
% mU_nL=zeros(k,kmeans_nb_vec);
% for i=1:k
%     mU_nL(i,:)=mean(U_nL(sum(cs(1:(i-1)))*n+1:sum(cs(1:i))*n,1:kmeans_nb_vec));
% end
% kmeans_output = kmeans(U_nL(:,1:kmeans_nb_vec),k,'Start',mU_nL);
% 
% tmp=0;
% for perm=perms(1:k)'
%     vec=zeros(n,1);
%     for i=1:k
%         vec(sum(cs(1:(i-1)))*n+1:sum(cs(1:i))*n)=perm(i)*ones(cs(i)*n,1);
%     end
%     if kmeans_output'*vec>tmp
%         tmp=kmeans_output'*vec;
%         best_vec=vec;
%     end
% end
% 
% kmeans_perf_KB = sum(best_vec==kmeans_output)/n;
%     
     store_perf(data_index,1) = kmeans_perf;
end



%alpha = ( norm_mu2^4 + 2*norm_mu2^3 + norm_mu2^2*(1-c*ratio) -2*c*norm_mu2 -c )/norm_mu2/(1+norm_mu2)^3;
%store_output(s_index,:) = [mean(store_perf), erfc(sqrt(alpha/2/(1-alpha)))/2, mean(store_align), alpha];

store_output(1:2) = [mean(store_perf), std(store_perf)];
%store_output(3) = mean(store_time(:,1));


%figure
%hold on
store_output
toc
%errorbar(s_range, store_output(:,1), store_output(:,3))
%errorbar(s_range, store_output(:,2), store_output(:,4))
%ylabel('misclassificaiton rate')
%legend(f_choice, 'uniform')
%xlabel('s')

%%
%figure

%hold on
%plot(s_range, store_output(:,6)/max(store_output(:,6)))
%plot(s_range, store_output(:,7)/max(store_output(:,7)))
%plot(s_range, store_output(:,5),'x')
%legend( ['compute time of ',f_choice], 'compute time of uniform', 'Proportion of non-zero')
%xlabel('s')
