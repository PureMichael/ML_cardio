clear all
close all
clc
%%%% two spirals
% t=0:0.01:2*pi;
% rn=1;
% rd=1;
% rnd=20;
% thetam=1;
% x=(rn./(t+rd)).*cos(thetam*t) + ((rand(length(t),1)-0.5)/rnd)';y=(rn./(t+rd)).*sin(thetam*t) + ((rand(length(t),1)-0.5)/rnd)';
% x2=(-rn./(t+rd)).*cos(-thetam*t) + ((rand(length(t),1)-0.5)/rnd)';y2=(rn./(t+rd)).*sin(-thetam*t) + ((rand(length(t),1)-0.5)/rnd)';
% plot(x,y,'b.','markersize',20);hold on;plot(x2,y2,'r.','markersize',20)
% xyl=[[x';x2'] [y';y2'] [zeros(length(x2),1);ones(length(y2),1)]];

%%%% many wiggly radial lines 
% t=pi/2:0.01:4*pi;
% rnd=10;
% sf=1/(2*pi);
% sf2=2;
% hold on
% for theta = 0:pi/4:(2*pi - pi/4)
%     x=sf*(t*cos(theta)-sf2*sin(t)*sin(theta));
%     y=sf*(t*sin(theta)+sf2*sin(t)*cos(theta));
%     x=x.*exp(t/3);
%     
%     x=x+((rand(length(t),1)-0.5)/rnd)';
%     y=y+((rand(length(t),1)-0.5)/rnd)';
%     plot(x,y,'.','markersize',20)
% end
% axis tight 
% box on

% % %  
t=pi/1.1:0.01:2*pi;
rnd=100;
sf=1/(2*pi);
phi=t/12;
sf2=2;
hold on
for theta = 0:pi/4:(2*pi - pi/4)
% for theta = 0 : 0 
    x=sf*(t*cos(theta)-sf2*sin(t)*sin(theta));
    y=sf*(t*sin(theta)+sf2*sin(t)*cos(theta));
    x=cos(phi).*x-sin(phi).*y;
    y=sin(phi).*x + cos(phi).*y;
    x=x+((rand(length(t),1)-0.5)/rnd)';
    y=y+((rand(length(t),1)-0.5)/rnd)';
    plot(x,y,'.','markersize',20)
end
axis tight 
box on



