
function [A, W, b, B] = SGRFME(X_train, Y_train,X_test,Y_test, num_labeled,X_Anchors, param)
diary on
warning off
%% Simultaneous anchor-data graph and RFME algo
% Input
%         X_train :: matrix containing the main data,
%                 the first num_labeled samples are considered as labeled
%         Y_train :: matrix containing the binary labels,
%                 the first num_labeled samples are considered as labeled
%		  X_test   :: matrix containing the test data
%		  Y_test   :: the label vector of test
%         X_Anchors :: Data matrix of anchors
%         num_labeled :: number of labeled samples
%         param.
%          k :: number of neibours for the B graph
%          ul :: FME param, weight for labeled samples
%          miu :: FME param
%          gamma :: FME param
%          lambda :: Weight of distance between the anchors
%          rho :: Weight for the G matrix
% Output
%         A :: Soft label matrix
%         W :: Projection matrix
%         b :: bian vector
%         B :: Anchor-data graph
%

%% Main Algo


%% Parameter defining
num_samples = size(X_train,1);

num_anchors = param.num_anchors;
k = param.k;

%% Balance parameters (can be changed)
miu=10^9;%para(1); 
gamma=10^9;%para(2);
lambda=10^-24;%para(3);
rho=10^-15;%para(4);
Uvalue=10^9;%para(5);


%% Estimate Anchors


Z = X_Anchors';

%% Estimate B matrix


%initialize weighted_distX
distX =  L2_distance_1( X_Anchors' , X_train' )' ;
[distXs, idx] = sort(distX,2);


B = zeros(num_samples , num_anchors);

for i = 1:num_samples
    id = idx(i,2:k+2);
    di = distXs(i,2:k+2);
    B(i,id) = (di(k+1)-di)/(k*di(k+1)-sum(di(1:k))+eps);             
        
end

clear id rr di distX distXs idx i num_clusters



%% Iterate

%% Some precalculations
Wtilde = B' * B;
Wtildes{1,1}=Wtilde;

Ha = eye(num_anchors) - (1/num_anchors) * ones(num_anchors,num_anchors);


dist_anchors =  L2_distance_1( X_Anchors' , X_Anchors')' ;

[ ~ , YB ] = max(Y_train');



%% Main loop
% U = sparse(num_samples,num_samples);
% for i = 1 : num_labeled
%     U(i,i) = 10;
% end
% clear i
% BtUB = B' * U * B ;
new_B=B(1:num_labeled,:);
BtUB=new_B'*new_B*Uvalue;
% BtUY = B' * U * Y_train;
BtUY=new_B'*Y_train(1:num_labeled,:)*Uvalue;

% diff=BtUY-BtUBY_new;
% diff=BTUB_new-BtUB;


            HaZtZHaZtgaminv = Ha * Z' * ( Z * Ha * Z' + gamma * eye(size(Z,1)) )^(-1) * Z * Ha ;
            
            

                    Wtilde = B' * B;
                    Wtildes{1,1}=Wtilde;

                    for iter = 1 : 10
                        %% Graph fixed, estiamte A,W,b
                        % RFME algo


                        Dtildemhalf = diag(sum(Wtilde).^(-0.5));
                        Ltilde = eye(num_anchors) - Dtildemhalf * Wtilde * Dtildemhalf;


                        %Eq7

                        A = (Ltilde + BtUB + miu * Ha - miu * HaZtZHaZtgaminv )^(-1)* BtUY;

                        %Eq8
                        W = (Z * Ha * Z' + gamma*eye(size(Z,1)))^(-1) * Z * Ha * A ;

                        %Eq9
                        b = (1/num_anchors)*(A'*ones(num_anchors,1) - W'*Z*ones(num_anchors,1));

                        % visualization of results
                         F = B*A;

                        [ ~ , FB] = max(F');
                         lbldiff = (FB-YB);
                         
                         acc_train(iter) =  length(find(lbldiff(num_labeled:end)==0)) /  length(lbldiff(num_labeled:end) );
                         


                        FTEST=(W' * X_test)+ b;
                        [ ~ , FT] = max(FTEST);
                        clear FTEST
                        lbltdiff = (FT-Y_test');
                        
                        acc_test(iter) =  length(find(lbltdiff(1:end)==0)) /  length(lbltdiff(1:end) );
                        

                %Validation       

                %                         FTRAIN=(W' * X_train')+ b;
                %                         [ ~ , FT] = max(FTRAIN);
                %                         lbltdiff = (FT-YB);
                %                         acc_trainlbl(iter) =  length(find(lbltdiff(num_labeled:end)==0)) /  length(lbltdiff(num_labeled:end) );
                %                         
                        clear Wtilde Dtilde Ltilde 

                        %% A,W,b fixed, estimate Wt


                        dist_A =  L2_distance_1( A' , A' )' ;
                        G = dist_A + lambda * dist_anchors;


                         Wtilde = zeros(num_anchors,num_anchors);
                        
                        Wtilde = MLAN_Solution(G/(2*rho),Wtilde,k);


                        % normalize Wtilde and make it symetric

                        Wtilde = (Wtilde+Wtilde')./2;

                        Wtildes{iter+1}=Wtilde;
                        % if the distance between two consecutive matrices is less than threshold, break the iteration
                        if sum(sum(abs(Wtildes{iter} - Wtildes{iter+1})))/sum(Wtildes{iter}(:)) < 0.01
                            break
                        end
                    end
                    

                   
                     disp(['miu=' num2str(miu)  ',gamma =' num2str(gamma)...
                        ',lambda =' num2str(lambda) ',rho =' num2str(rho) ',UValue =' num2str(Uvalue) ' accuracy train unlabel ' num2str(100*acc_train)])
                    disp(['miu=' num2str(miu)  ',gamma =' num2str(gamma)...
                        ',lambda =' num2str(lambda) ',rho =' num2str(rho) ',UValue =' num2str(Uvalue) ' accuracy test ' num2str(100*acc_test)])
                    % disp(['accuracy train label ' num2str(100*acc_trainlbl)])
          
                   disp('----------------------------------------------------')

                    clear acc_test acc_train Dtildemhalf Wtilde Ltilde F FB lbldiff dist_A G Wtildes C 

end















