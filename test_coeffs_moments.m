clear all;
tau = 0.2874;
options1 = optimset('Display','iter','TolX',1e-8,'MaxFunEvals',10000,'MaxIter',10000);
piecewise_coeff = fsolve(@(x) find_coeff_piecewise2(x, [1/2, 1/(sqrt(8*pi*tau)), tau*(1/2-1/(2*pi))], tau), .5*randn(1,3), options1)
find_coeff_piecewise2(piecewise_coeff, [1/2, 1/(sqrt(8*pi*tau)), tau*(1/2-1/(2*pi))], tau)