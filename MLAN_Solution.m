

function S = MLAN_Solution(G,S,k)



[distXs, idx] = sort(G,2);

[num_samples , num_anchors]= size(S);

for i = 1:num_samples
    idx0 = idx(i,2:k+2);

        ad = -G(i,idx0);
        S(i,idx0) = EProjSimplex_new(ad);

end