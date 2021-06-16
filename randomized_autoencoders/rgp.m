function [g,r] = rgp(x,y,k,varargin)
  running_choice = varargin{1}.running_choice;
  %f_choice = varargin{1}.f_choice;

  switch running_choice
    %case 'RMT'
    %       switch f_choice
    %           case 'lin'
    %               f = @(t) t;
    %           case 'sparse'
    %               f = @(t) t.*(1 + sign(abs(t)-sqrt(2)*s))/2;
    %           case 'sign'
    %               s=varargin{1}.s;
    %               f = @(t) sign(t).*(1 + sign(abs(t)-sqrt(2)*s))/2;
    %               z = f(x);
    %           case 'relu'
    %               f = @(t) max(t,0);
    %               z = f(x);
    %           case 'abs'
    %               f = @(t) abs(t);
    %               z = f(x);
    %           case 'binary'
    %               f = @(x, s) (x>(sqrt(2)*s)) + (x<(-sqrt(2)*s));
    %               s = 0.5; z = f(x,s);
    %           case 'ternary'
    %               f = @(x, s_minus, s_plus, r, t) (t*((x>(sqrt(2)*s_plus))) - r*t*(x<(sqrt(2)*s_minus)));
    %               s_minus=varargin{1}.s_minus; s_plus=varargin{1}.s_plus; r=varargin{1}.r; t=varargin{1}.t;
    %               z = f(x, s_minus, s_plus, r, t);
    %           case 'quant'
    %               M = 2;
    %               if s == 0
    %                   f = @(t) sign(t);
    %               else
    %                   f = @(x) (floor(x*2^(M-2)/sqrt(2)/s) + 1/2)/(2^(M-2)).*(abs(x)<=sqrt(2)*s) + sign(x).*(abs(x)>sqrt(2)*s);
    %               end
    %               z = f(x);
    %           case 'rcca'
    %               f      = @(x0) cos(bsxfun(x0*w));
    %       end
    case 'RMT'
      f       = aug(x,k, varargin);
      z       = f(x);
    case 'RCCA'
      f       = aug(x,k, varargin);
      z       = f(x);
  end
  a = inv((eye(size(z,2))*1e-6+z'*z))*(z'*y);
  g = @(x0) f(x0)*a;
  r = g(x);

end
