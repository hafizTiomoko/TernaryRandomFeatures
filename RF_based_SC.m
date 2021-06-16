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
		
	case 'CIFAR'
        selected_labels=[0 1];
        noise_level_dB= -Inf; %-Inf, -3 or 3
        data1 = readNPY('../datasets/CIFAR_feature/Real_hamburger_googlenet.npy');
        data2 = readNPY('../datasets/CIFAR_feature/Real_coffee_googlenet.npy');
%         data1 = readNPY('../datasets/CIFAR_feature/Real_pizza_googlenet.npy');
%         data2 = readNPY('../datasets/CIFAR_feature/Real_daisy_googlenet.npy');
end

if length(selected_labels) ~= k
    error('Error: selected labels and nb of classes not equal!')
end


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
            f = @(x, s) (abs(x)>(sqrt(2)*s));
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

noise_level=10^(noise_level_dB/10);
Noise = rand(p,n)*sqrt(12)*sqrt(noise_level*var(X(:)));
X = X*(eye(n) - ones(n)/n);
X=X+Noise;

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


    W = sign(randn(N,p));  %%% Symmetric Bernoulli weights
    %W = randn(N,p);		%%% Gaussian weights
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

switch f_choice
    case 'sparse'
        prop_zero = sum(sum(K==0))/n^2;
        eps_unif = 1-prop_zero;
    case 'sign'
        prop_zero = sum(sum(f(W*X)==0))/n^2;
        %eps_unif = (1-prop_zero)/64;
        store_output(3) = (1-prop_zero)/64;
    case 'relu'
        prop_zero = sum(sum(f(W*X)==0))/n^2
        store_output(3) = (1-prop_zero);
    case 'ternary'
        prop_zero = sum(sum(f(W*X, s_minus, s_plus, r, t)==0))/n^2
        store_output(3) = (1-prop_zero)/64;
    case 'abs'
        prop_zero = sum(sum(f(W*X)==0))/n^2
        store_output(3) = (1-prop_zero)/64;
    case 'binary'
        prop_zero = sum(sum(f(W*X, s)==0))/n^2
        store_output(3) = (1-prop_zero)/64;

end


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

	noise_level=10^(noise_level_dB/10);
	Noise = rand(p,n)*sqrt(12)*sqrt(noise_level*var(X(:)));
	X = X*(eye(n) - ones(n)/n);
	X=X+Noise;

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


    store_perf(data_index,1) = kmeans_perf;
end



store_output(1:2) = [mean(store_perf), std(store_perf)];

store_output
toc
