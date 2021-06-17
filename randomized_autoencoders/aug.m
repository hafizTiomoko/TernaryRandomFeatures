function f = aug(x,k, varargin)
 sub    = 1000;
 perm   = randperm(size(x,1));
 sample = x(perm(1:sub),:);

 norms  = sum(sample.^2,2);
 dist   = norms*ones(1,sub)+ones(sub,1)*norms'-2*sample*sample';
 s      = 1/median(dist(:));
 w      = 2*s*randn(size(x,2),k);
 b      = 2*pi*rand(1,k)-pi;
 
 f_choice = varargin{1}{1}.f_choice;
switch f_choice
  case 'lin'
      f  = @(x0) bsxfun(@plus,x0*w,b);
  case 'sign'
      %s=varargin{1}{1}.s;
      %f = @(t) sign(fn(t)).*(1 + sign(abs(fn(t))-sqrt(2)*s))/2;
      w      = randn(size(x,2),k)/sqrt(size(x,2));
      f = @(x0) sign(x0*w);
  case 'relu'
      %w      = randn(size(x,2),k)/sqrt(size(x,2));
      %f  = @(x0) max(bsxfun(@plus,x0*w,b),0);
      f = @(x0) max(x0*w,0);
  case 'abs'
      %w      = randn(size(x,2),k)/sqrt(size(x,2));
      %f  = @(x0) abs(bsxfun(@plus,x0*w,b));
      f = @(x0) abs(x0*w);
  case 'ternary'
      %w      = randn(size(x,2),k)/sqrt(size(x,2));
      s_minus = varargin{1}{1}.s_minus;
      s_plus = varargin{1}{1}.s_plus;
      r = varargin{1}{1}.r;
      t = varargin{1}{1}.t;
      f = @(x0) (t*(((x0*w)>(sqrt(2)*s_plus))) - r*t*((x0*w)<(sqrt(2)*s_minus)));
   
  case 'binary'
      s_plus = varargin{1}{1}.s;
      %f = @(x0) max(max(abs(x0*w)));
      f = @(x0) ((abs(x0*w)>(sqrt(2)*s_plus)));
  case 'rcca'
      %w      = randn(size(x,2),k)/sqrt(size(x,2));
      f = @(x0) cos(x0*w);
      %f  = @(x0) cos(bsxfun(@plus,x0*w,b));
end
