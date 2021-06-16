function F = find_coeff_piecewise2(x,coeff, tau) %%% x = [s_minus, s_plus, t]; coeff = [a1, a2, nu]
r = @(x) (1-erf(x(2)/sqrt(tau)))/(1+erf(x(1)/sqrt(tau)));
F(1) =x(3)*(exp(-x(2)*x(2)/tau) + r(x)*exp(-x(1)*x(1)/tau))/sqrt(2*pi*tau) - coeff(1);
F(2) = x(3)*(x(2)*exp(-x(2)*x(2)/tau) + r(x)*x(1)*exp(-x(1)*x(1)/tau))/(2*tau*sqrt(pi*tau)) - coeff(2);
F(3) = x(3)*x(3)*((1-erf(x(2)/sqrt(tau)))*(1+r(x))/2) - coeff(3);
%F(1) = (exp(-x(2)*x(2)/tau) + r(x)*exp(-x(1)*x(1)/tau) )/sqrt(2*pi*tau) - coeff(1);
%F(2) = (x(2)*exp(-x(2)*x(2)/tau) + r(x)*x(1)*exp(-x(1)*x(1))/tau)/(tau*sqrt(4*pi*tau)) - coeff(2);
%F(2) = ((1-erf(sqrt(2)*x(2)/sqrt(tau)))*(1+r(x))/2- (exp(-x(2)*x(2)/tau) + r(x)*exp(-x(1)*x(1)/tau))^2/(2*pi)) -coeff(2);
end