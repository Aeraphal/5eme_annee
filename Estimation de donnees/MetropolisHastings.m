function [x,alpha,beta] = MetropolisHastings(T,alpha,beta,sigmaq)
    

    %Création de la série de variables aléatoires identiquement distibués 
    x = rand([1,T]);
    %Création de la loi normale centrée réduite.
    norm_centr_red = normpdf(x,0,1);
    
    for k=1:T
        x(1,k) = x(1,k) + sigmaq*norm_centr_red(1,k);
    
    end
    
    %Affichage de la fonction
    figure(1);
    plot(norm_centr_red),"Loi normale centrée réduite";
    
end