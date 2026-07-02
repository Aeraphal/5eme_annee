clear variables;
close all;

%programmer 2 fonctions

%Une fonction proposition
%Une fonction qui va faire un tirage suivant la loi de proposition, indépendant de tous le reste

%une loi cible
%une fonction qui pour un x donnée, renvoie la loi de probabilité cible de la loi donnée

alpha = 0.5;
beta = 0.5;
T = 10000;
sigmaq = 1;
x = rand(1,N);
x_t = [0:1/T;1];
Beta = gamma(a)*gamma(b)/gamma(a+b);
f_x_t = betapdf(x,a,b);



