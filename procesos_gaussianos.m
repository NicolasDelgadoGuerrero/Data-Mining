%Creamos cada uno de los Folds con sus respectivos conjuntos train y test.
%Fold1: 1Test y 4Train.
Fold1_Test = [Malign_folds(1).histogram; Healthy_folds(1).histogram];

Fold1_Train_1 = [Malign_folds(2).histogram; Malign_folds(3).histogram;...
                 Malign_folds(4).histogram; Malign_folds(5).histogram;...
                 Healthy_folds(2).histogram];
                 
Fold1_Train_2 = [Malign_folds(2).histogram; Malign_folds(3).histogram;...
                 Malign_folds(4).histogram; Malign_folds(5).histogram;...
                 Healthy_folds(3).histogram];
                 
Fold1_Train_3 = [Malign_folds(2).histogram; Malign_folds(3).histogram;...
                 Malign_folds(4).histogram; Malign_folds(5).histogram;...
                 Healthy_folds(4).histogram];                 
                 
Fold1_Train_4 = [Malign_folds(2).histogram; Malign_folds(3).histogram;...
                 Malign_folds(4).histogram; Malign_folds(5).histogram;...
                 Healthy_folds(5).histogram];
                 
%Fold2: 1Test y 4Train.
Fold2_Test = [Malign_folds(2).histogram; Healthy_folds(2).histogram];

Fold2_Train_1 = [Malign_folds(1).histogram; Malign_folds(3).histogram; Malign_folds(4).histogram; Malign_folds(5).histogram; Healthy_folds(1).histogram];
                 
Fold2_Train_2 = [Malign_folds(1).histogram; Malign_folds(3).histogram; Malign_folds(4).histogram; Malign_folds(5).histogram; Healthy_folds(3).histogram];
                 
Fold2_Train_3 = [Malign_folds(1).histogram; Malign_folds(3).histogram; Malign_folds(4).histogram; Malign_folds(5).histogram; Healthy_folds(4).histogram];                 
                 
Fold2_Train_4 = [Malign_folds(1).histogram; Malign_folds(3).histogram; Malign_folds(4).histogram; Malign_folds(5).histogram; Healthy_folds(5).histogram];

%Fold3: 1Test y 4Train.
Fold3_Test = [Malign_folds(3).histogram; Healthy_folds(3).histogram];

Fold3_Train_1 = [Malign_folds(1).histogram; Malign_folds(2).histogram; Malign_folds(4).histogram; Malign_folds(5).histogram; Healthy_folds(1).histogram];
                 
Fold3_Train_2 = [Malign_folds(1).histogram; Malign_folds(2).histogram; Malign_folds(4).histogram; Malign_folds(5).histogram; Healthy_folds(2).histogram];
                 
Fold3_Train_3 = [Malign_folds(1).histogram; Malign_folds(2).histogram; Malign_folds(4).histogram; Malign_folds(5).histogram; Healthy_folds(4).histogram];                 
                 
Fold3_Train_4 = [Malign_folds(1).histogram; Malign_folds(2).histogram; Malign_folds(4).histogram; Malign_folds(5).histogram; Healthy_folds(5).histogram];              

%Fold4: 1Test y 4Train.
Fold4_Test = [Malign_folds(4).histogram; Healthy_folds(4).histogram];

Fold4_Train_1 = [Malign_folds(1).histogram; Malign_folds(2).histogram; Malign_folds(3).histogram; Malign_folds(5).histogram; Healthy_folds(1).histogram];
                 
Fold4_Train_2 = [Malign_folds(1).histogram; Malign_folds(2).histogram; Malign_folds(3).histogram; Malign_folds(5).histogram; Healthy_folds(2).histogram];
                 
Fold4_Train_3 = [Malign_folds(1).histogram; Malign_folds(2).histogram; Malign_folds(3).histogram; Malign_folds(5).histogram; Healthy_folds(3).histogram];                 
                 
Fold4_Train_4 = [Malign_folds(1).histogram; Malign_folds(2).histogram; Malign_folds(3).histogram; Malign_folds(5).histogram; Healthy_folds(5).histogram];    
       
%Fold5: 1Test y 4Train.
Fold5_Test = [Malign_folds(5).histogram; Healthy_folds(5).histogram];

Fold5_Train_1 = [Malign_folds(1).histogram; Malign_folds(2).histogram; Malign_folds(3).histogram; Malign_folds(4).histogram; Healthy_folds(1).histogram];
                 
Fold5_Train_2 = [Malign_folds(1).histogram; Malign_folds(2).histogram; Malign_folds(3).histogram; Malign_folds(4).histogram; Healthy_folds(2).histogram];
                 
Fold5_Train_3 = [Malign_folds(1).histogram; Malign_folds(2).histogram; Malign_folds(3).histogram; Malign_folds(4).histogram; Healthy_folds(3).histogram];                 
                 
Fold5_Train_4 = [Malign_folds(1).histogram; Malign_folds(2).histogram; Malign_folds(3).histogram; Malign_folds(4).histogram; Healthy_folds(4).histogram];       
                 
%Comprobamos dimensiones de los conjuntos de entrenamiento y test
[Tam_Test_Fold1 nada] = size(Fold1_Test);
[Tam_Train_Fold1 nada] = size([Fold1_Train_1; Fold1_Train_2; Fold1_Train_3; Fold1_Train_4]);           

[Tam_Test_Fold2 nada] = size(Fold2_Test);
[Tam_Train_Fold2 nada] = size([Fold2_Train_1; Fold2_Train_2; Fold2_Train_3; Fold2_Train_4]);  

[Tam_Test_Fold3 nada] = size(Fold3_Test);
[Tam_Train_Fold3 nada] = size([Fold3_Train_1; Fold3_Train_2; Fold3_Train_3; Fold3_Train_4]);

[Tam_Test_Fold4 nada] = size(Fold4_Test);
[Tam_Train_Fold4 nada] = size([Fold4_Train_1; Fold4_Train_2; Fold4_Train_3; Fold4_Train_4]);
[Tam_Test_Fold5 nada] = size(Fold5_Test);
[Tam_Train_Fold5 nada] = size([Fold5_Train_1; Fold5_Train_2; Fold5_Train_3; Fold5_Train_4]);

Tam_Test_Fold1
Tam_Train_Fold1
Tam_Test_Fold2
Tam_Train_Fold2
Tam_Test_Fold3
Tam_Train_Fold3
Tam_Test_Fold4
Tam_Train_Fold4
Tam_Test_Fold5
Tam_Train_Fold5

%Realizamos procesos gaussianos para el Fold1
%Medias y Covarainzas necesarias para definir un proceso Gaussiano
%Media cero:
meanfunc = @meanZero;

%Matriz de covarianzas con kernel radial
%Definimos parámetros escala y varianza
ell = 1.9;
sf = 1.0;
hyp.cov = log([ell sf]);
covfunc = @covSEiso;
%covfunc = @covLIN;

%Añadimos etiquetas: 1=tejido cancerígeno -1=tejido no canrcerígeno
Fold1_Test_Labels = [ones(54,1); -ones(203,1)];
Fold1_Train_1_Labels = [ones(244,1); -ones(210,1)];
Fold1_Train_2_Labels = [ones(244,1); -ones(206,1)];
Fold1_Train_3_Labels = [ones(244,1); -ones(196,1)];
Fold1_Train_4_Labels = [ones(244,1); -ones(199,1)];

%Modelo de observación: Regresión logística
likfunc = @likLogistic;

%Estimamos parámetros de nuestro modelo a priori (ell, sf)
hyp_1 = minimize(hyp, @gp, -40, @infVB, meanfunc, covfunc, likfunc, Fold1_Train_1, Fold1_Train_1_Labels);
%hyp_1 = [];

%Calculamos la probabilidad de pertenecer a una clase mediante un proceso Gaussiano             
[a b c d lp] = gp(hyp_1, @infVB, meanfunc, covfunc, likfunc, Fold1_Train_1, Fold1_Train_1_Labels, Fold1_Test, ones(257,1));

prob_test_1=exp(lp);               

%Realizamos el mismo proceso para cada uno de los train de Fold1                 
hyp_2 = minimize(hyp, @gp, -40, @infVB, meanfunc, covfunc, likfunc, Fold1_Train_2, Fold1_Train_2_Labels);
hyp_3 = minimize(hyp, @gp, -40, @infVB, meanfunc, covfunc, likfunc, Fold1_Train_3, Fold1_Train_3_Labels);
hyp_4 = minimize(hyp, @gp, -40, @infVB, meanfunc, covfunc, likfunc, Fold1_Train_4, Fold1_Train_4_Labels);
%hyp_2 = [];
%hyp_3 = [];
%hyp_4 = [];

[a b c d lp] = gp(hyp_2, @infVB, meanfunc, covfunc, likfunc, Fold1_Train_2, Fold1_Train_2_Labels, Fold1_Test, ones(257,1));
prob_test_2=exp(lp);  

[a b c d lp] = gp(hyp_3, @infVB, meanfunc, covfunc, likfunc, Fold1_Train_3, Fold1_Train_3_Labels, Fold1_Test, ones(257,1));
prob_test_3=exp(lp);

[a b c d lp] = gp(hyp_4, @infVB, meanfunc, covfunc, likfunc, Fold1_Train_4, Fold1_Train_4_Labels, Fold1_Test, ones(257,1));
prob_test_4=exp(lp);

%Calculamos la probabilidad media de pertenecer a la clase 1 y las curvas ROC y
%precisión respectivamente
prob_mean_Fold1 = (prob_test_1 + prob_test_2 + prob_test_3 + prob_test_4)/4;

[X1, Y1, T, AUC1] = perfcurve(Fold1_Test_Labels, prob_mean_Fold1, 1);

[X2, Y2, T2pr, AUC2] = perfcurve(Fold1_Test_Labels, prob_mean_Fold1, 1, 'xCrit', 'sens', 'yCrit', 'prec'); 

figure, plot(X1,Y1), xlabel('False positive rate'), ylabel('True positive rate'), title('ROC for Classification by Logistic Regression Fold1')
figure, plot(X2, Y2), xlabel('Sensitive rate'), ylabel('Precision rate'), title('ROC for Classification by Logistic Regression Fold1')

AUC1
AUC2

%Hacemos las predicciones fijando una threshold ?=0.5
prediciones = ones(257, 1);
prediciones(prob_mean_Fold1 <= 0.5) = -1;

%Calculamos matriz de confusión y métricas de bondad del clasificador
Confusion_Fold1 = confusionmat(Fold1_Test_Labels, prediciones, 'Order',[1 -1])
accuracy_Fold1 = sum(diag(Confusion_Fold1)) / sum(sum(Confusion_Fold1))
specificity_Fold1 = Confusion_Fold1(2,2) / sum(Confusion_Fold1(2,:))
sensitivity_Fold1 = Confusion_Fold1(1,1) / sum(Confusion_Fold1(1,:))
precision_Fold1 = Confusion_Fold1(1,1) / sum(Confusion_Fold1(:,1))
F_score_Fold1 = (2*precision_Fold1 * sensitivity_Fold1) / (precision_Fold1 + sensitivity_Fold1)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Hacemos el mismo proceso para cada uno de los folds 

%Realizamos procesos gaussianos para el Fold2
%Medias y Covarainzas necesarias para definir un proceso Gaussiano
%Media cero:
meanfunc = @meanZero;

%Matriz de covarianzas con kernel radial
%Definimos parámetros escala y varianza
ell = 1.9;
sf = 1.0;
%hyp.cov = log([ell sf]);
covfunc = @covSEiso;
%covfunc = @covLIN;

%Añadimos etiquetas: 1=tejido cancerígeno -1=tejido no canrcerígeno
Fold2_Test_Labels = [ones(72,1); -ones(210,1)];
Fold2_Train_1_Labels = [ones(226,1); -ones(203,1)];
Fold2_Train_2_Labels = [ones(226,1); -ones(206,1)];
Fold2_Train_3_Labels = [ones(226,1); -ones(196,1)];
Fold2_Train_4_Labels = [ones(226,1); -ones(199,1)];

%Modelo de observación: Regresión logística
likfunc = @likLogistic;

%Estimamos parámetros de nuestro modelo a priori (ell, sf)
hyp_1 = minimize(hyp, @gp, -40, @infVB, meanfunc, covfunc, likfunc, Fold2_Train_1, Fold2_Train_1_Labels);
%hyp_1 = [];

%Calculamos la probabilidad de pertenecer a una clase mediante un proceso Gaussiano             
[a b c d lp] = gp(hyp_1, @infVB, meanfunc, covfunc, likfunc, Fold2_Train_1, Fold2_Train_1_Labels, Fold2_Test, ones(282,1));

prob_test_1=exp(lp);               

%Realizamos el mismo proceso para cada uno de los train de Fold1                 
hyp_2 = minimize(hyp, @gp, -40, @infVB, meanfunc, covfunc, likfunc, Fold2_Train_2, Fold2_Train_2_Labels);
hyp_3 = minimize(hyp, @gp, -40, @infVB, meanfunc, covfunc, likfunc, Fold2_Train_3, Fold2_Train_3_Labels);
hyp_4 = minimize(hyp, @gp, -40, @infVB, meanfunc, covfunc, likfunc, Fold2_Train_4, Fold2_Train_4_Labels);
%hyp_2 = [];
%hyp_3 = [];
%hyp_4 = [];

[a b c d lp] = gp(hyp_2, @infVB, meanfunc, covfunc, likfunc, Fold2_Train_2, Fold2_Train_2_Labels, Fold2_Test, ones(282,1));
prob_test_2=exp(lp);  

[a b c d lp] = gp(hyp_3, @infVB, meanfunc, covfunc, likfunc, Fold2_Train_3, Fold2_Train_3_Labels, Fold2_Test, ones(282,1));
prob_test_3=exp(lp);

[a b c d lp] = gp(hyp_4, @infVB, meanfunc, covfunc, likfunc, Fold2_Train_4, Fold2_Train_4_Labels, Fold2_Test, ones(282,1));
prob_test_4=exp(lp);

%Calculamos la probabilidad media de pertenecer a la clase 1 y las curvas ROC y
%precisión respectivamente
prob_mean_Fold2 = (prob_test_1 + prob_test_2 + prob_test_3 + prob_test_4)/4;

[X1, Y1, T, AUC1] = perfcurve(Fold2_Test_Labels, prob_mean_Fold2, 1);

[X2, Y2, T2pr, AUC2] = perfcurve(Fold2_Test_Labels, prob_mean_Fold2, 1, 'xCrit', 'sens', 'yCrit', 'prec'); 

figure, plot(X1,Y1), xlabel('False positive rate'), ylabel('True positive rate'), title('ROC for Classification by Logistic Regression Fold2')
figure, plot(X2, Y2), xlabel('Sensitive rate'), ylabel('Precision rate'), title('ROC for Classification by Logistic Regression Fold2')

AUC1
AUC2

%Hacemos las predicciones fijando una threshold ?=0.5
prediciones = ones(282, 1);
prediciones(prob_mean_Fold2 <= 0.5) = -1;

%Calculamos matriz de confusión y métricas de bondad del clasificador
Confusion_Fold2 = confusionmat(Fold2_Test_Labels, prediciones, 'Order',[1 -1])
accuracy_Fold2 = sum(diag(Confusion_Fold2)) / sum(sum(Confusion_Fold2))
specificity_Fold2 = Confusion_Fold2(2,2) / sum(Confusion_Fold2(2,:))
sensitivity_Fold2 = Confusion_Fold2(1,1) / sum(Confusion_Fold2(1,:))
precision_Fold2 = Confusion_Fold2(1,1) / sum(Confusion_Fold2(:,1))
F_score_Fold2 = (2*precision_Fold2 * sensitivity_Fold2) / (precision_Fold2 + sensitivity_Fold2)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Realizamos procesos gaussianos para el Fold3
%Medias y Covarainzas necesarias para definir un proceso Gaussiano
%Media cero:
meanfunc = @meanZero;

%Matriz de covarianzas con kernel radial
%Definimos parámetros escala y varianza
ell = 1.9;
sf = 1.0;
hyp.cov = log([ell sf]);
covfunc = @covSEiso;
%covfunc = @covLIN;

%Añadimos etiquetas: 1=tejido cancerígeno -1=tejido no canrcerígeno
Fold3_Test_Labels = [ones(53,1); -ones(206,1)];
Fold3_Train_1_Labels = [ones(245,1); -ones(203,1)];
Fold3_Train_2_Labels = [ones(245,1); -ones(210,1)];
Fold3_Train_3_Labels = [ones(245,1); -ones(196,1)];
Fold3_Train_4_Labels = [ones(245,1); -ones(199,1)];

%Modelo de observación: Regresión logística
likfunc = @likLogistic;

%Estimamos parámetros de nuestro modelo a priori (ell, sf)
hyp_1 = minimize(hyp, @gp, -40, @infVB, meanfunc, covfunc, likfunc, Fold3_Train_1, Fold3_Train_1_Labels);
%hyp_1 = [];

%Calculamos la probabilidad de pertenecer a una clase mediante un proceso Gaussiano             
[a b c d lp] = gp(hyp_1, @infVB, meanfunc, covfunc, likfunc, Fold3_Train_1, Fold3_Train_1_Labels, Fold3_Test, ones(259,1));

prob_test_1=exp(lp);               

%Realizamos el mismo proceso para cada uno de los train de Fold1                 
hyp_2 = minimize(hyp, @gp, -40, @infVB, meanfunc, covfunc, likfunc, Fold3_Train_2, Fold3_Train_2_Labels);
hyp_3 = minimize(hyp, @gp, -40, @infVB, meanfunc, covfunc, likfunc, Fold3_Train_3, Fold3_Train_3_Labels);
hyp_4 = minimize(hyp, @gp, -40, @infVB, meanfunc, covfunc, likfunc, Fold3_Train_4, Fold3_Train_4_Labels);
%hyp_2 = [];
%hyp_3 = [];
%hyp_4 = [];

[a b c d lp] = gp(hyp_2, @infVB, meanfunc, covfunc, likfunc, Fold3_Train_2, Fold3_Train_2_Labels, Fold3_Test, ones(259,1));
prob_test_2=exp(lp);  

[a b c d lp] = gp(hyp_3, @infVB, meanfunc, covfunc, likfunc, Fold3_Train_3, Fold3_Train_3_Labels, Fold3_Test, ones(259,1));
prob_test_3=exp(lp);

[a b c d lp] = gp(hyp_4, @infVB, meanfunc, covfunc, likfunc, Fold3_Train_4, Fold3_Train_4_Labels, Fold3_Test, ones(259,1));
prob_test_4=exp(lp);

%Calculamos la probabilidad media de pertenecer a la clase 1 y las curvas ROC y
%precisión respectivamente
prob_mean_Fold3 = (prob_test_1 + prob_test_2 + prob_test_3 + prob_test_4)/4;

[X1, Y1, T, AUC1] = perfcurve(Fold3_Test_Labels, prob_mean_Fold3, 1);

[X2, Y2, T2pr, AUC2] = perfcurve(Fold3_Test_Labels, prob_mean_Fold3, 1, 'xCrit', 'sens', 'yCrit', 'prec'); 

figure, plot(X1,Y1), xlabel('False positive rate'), ylabel('True positive rate'), title('ROC for Classification by Logistic Regression Fold')
figure, plot(X2, Y2), xlabel('Sensitive rate'), ylabel('Precision rate'), title('ROC for Classification by Logistic Regression Fold3')

AUC1
AUC2

%Hacemos las predicciones fijando una threshold ?=0.5
prediciones = ones(259, 1);
prediciones(prob_mean_Fold3 <= 0.5) = -1;

%Calculamos matriz de confusión y métricas de bondad del clasificador
Confusion_Fold3 = confusionmat(Fold3_Test_Labels, prediciones, 'Order',[1 -1])
accuracy_Fold3 = sum(diag(Confusion_Fold3)) / sum(sum(Confusion_Fold3))
specificity_Fold3 = Confusion_Fold3(2,2) / sum(Confusion_Fold3(2,:))
sensitivity_Fold3 = Confusion_Fold3(1,1) / sum(Confusion_Fold3(1,:))
precision_Fold3 = Confusion_Fold3(1,1) / sum(Confusion_Fold3(:,1))
F_score_Fold3 = (2*precision_Fold3 * sensitivity_Fold3) / (precision_Fold3 + sensitivity_Fold3)


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Realizamos procesos gaussianos para el Fold4
%Medias y Covarainzas necesarias para definir un proceso Gaussiano
%Media cero:
meanfunc = @meanZero;

%Matriz de covarianzas con kernel radial
%Definimos parámetros escala y varianza
ell = 1.9;
sf = 1.0;
hyp.cov = log([ell sf]);
covfunc = @covSEiso;
%covfunc = @covLIN;

%Añadimos etiquetas: 1=tejido cancerígeno -1=tejido no canrcerígeno
Fold4_Test_Labels = [ones(50,1); -ones(196,1)];
Fold4_Train_1_Labels = [ones(248,1); -ones(203,1)];
Fold4_Train_2_Labels = [ones(248,1); -ones(210,1)];
Fold4_Train_3_Labels = [ones(248,1); -ones(206,1)];
Fold4_Train_4_Labels = [ones(248,1); -ones(199,1)];

%Modelo de observación: Regresión logística
likfunc = @likLogistic;

%Estimamos parámetros de nuestro modelo a priori (ell, sf)
hyp_1 = minimize(hyp, @gp, -40, @infVB, meanfunc, covfunc, likfunc, Fold4_Train_1, Fold4_Train_1_Labels);
%hyp_1 = [];

%Calculamos la probabilidad de pertenecer a una clase mediante un proceso Gaussiano             
[a b c d lp] = gp(hyp_1, @infVB, meanfunc, covfunc, likfunc, Fold4_Train_1, Fold4_Train_1_Labels, Fold4_Test, ones(246,1));

prob_test_1=exp(lp);               

%Realizamos el mismo proceso para cada uno de los train de Fold1                 
hyp_2 = minimize(hyp, @gp, -40, @infVB, meanfunc, covfunc, likfunc, Fold4_Train_2, Fold4_Train_2_Labels);
hyp_3 = minimize(hyp, @gp, -40, @infVB, meanfunc, covfunc, likfunc, Fold4_Train_3, Fold4_Train_3_Labels);
hyp_4 = minimize(hyp, @gp, -40, @infVB, meanfunc, covfunc, likfunc, Fold4_Train_4, Fold4_Train_4_Labels);
%hyp_2 = [];
%hyp_3 = [];
%hyp_4 = [];

[a b c d lp] = gp(hyp_2, @infVB, meanfunc, covfunc, likfunc, Fold4_Train_2, Fold4_Train_2_Labels, Fold4_Test, ones(246,1));
prob_test_2=exp(lp);  

[a b c d lp] = gp(hyp_3, @infVB, meanfunc, covfunc, likfunc, Fold4_Train_3, Fold4_Train_3_Labels, Fold4_Test, ones(246,1));
prob_test_3=exp(lp);

[a b c d lp] = gp(hyp_4, @infVB, meanfunc, covfunc, likfunc, Fold4_Train_4, Fold4_Train_4_Labels, Fold4_Test, ones(246,1));
prob_test_4=exp(lp);

%Calculamos la probabilidad media de pertenecer a la clase 1 y las curvas ROC y
%precisión respectivamente
prob_mean_Fold4 = (prob_test_1 + prob_test_2 + prob_test_3 + prob_test_4)/4;

[X1, Y1, T, AUC1] = perfcurve(Fold4_Test_Labels, prob_mean_Fold4, 1);

[X2, Y2, T2pr, AUC2] = perfcurve(Fold4_Test_Labels, prob_mean_Fold4, 1, 'xCrit', 'sens', 'yCrit', 'prec'); 

figure, plot(X1,Y1), xlabel('False positive rate'), ylabel('True positive rate'), title('ROC for Classification by Logistic Regression Fold4')
figure, plot(X2, Y2), xlabel('Sensitive rate'), ylabel('Precision rate'), title('ROC for Classification by Logistic Regression Fold4')

AUC1
AUC2

%Hacemos las predicciones fijando una threshold ?=0.5
prediciones = ones(246, 1);
prediciones(prob_mean_Fold4 <= 0.5) = -1;

%Calculamos matriz de confusión y métricas de bondad del clasificador
Confusion_Fold4 = confusionmat(Fold4_Test_Labels, prediciones, 'Order',[1 -1])
accuracy_Fold4 = sum(diag(Confusion_Fold4)) / sum(sum(Confusion_Fold4))
specificity_Fold4 = Confusion_Fold4(2,2) / sum(Confusion_Fold4(2,:))
sensitivity_Fold4 = Confusion_Fold4(1,1) / sum(Confusion_Fold4(1,:))
precision_Fold4 = Confusion_Fold4(1,1) / sum(Confusion_Fold4(:,1))
F_score_Fold4 = (2*precision_Fold4 * sensitivity_Fold4) / (precision_Fold4 + sensitivity_Fold4)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Realizamos procesos gaussianos para el Fold5
%Medias y Covarainzas necesarias para definir un proceso Gaussiano
%Media cero:
meanfunc = @meanZero;

%Matriz de covarianzas con kernel radial
%Definimos parámetros escala y varianza
ell = 1.9;
sf = 1.0;
hyp.cov = log([ell sf]);
covfunc = @covSEiso;
%covfunc = @covLIN;

%Añadimos etiquetas: 1=tejido cancerígeno -1=tejido no canrcerígeno
Fold5_Test_Labels = [ones(69,1); -ones(199,1)];
Fold5_Train_1_Labels = [ones(229,1); -ones(203,1)];
Fold5_Train_2_Labels = [ones(229,1); -ones(210,1)];
Fold5_Train_3_Labels = [ones(229,1); -ones(206,1)];
Fold5_Train_4_Labels = [ones(229,1); -ones(196,1)];

%Modelo de observación: Regresión logística
likfunc = @likLogistic;

%Estimamos parámetros de nuestro modelo a priori (ell, sf)
hyp_1 = minimize(hyp, @gp, -40, @infVB, meanfunc, covfunc, likfunc, Fold5_Train_1, Fold5_Train_1_Labels);
%hyp_1 = [];

%Calculamos la probabilidad de pertenecer a una clase mediante un proceso Gaussiano             
[a b c d lp] = gp(hyp_1, @infVB, meanfunc, covfunc, likfunc, Fold5_Train_1, Fold5_Train_1_Labels, Fold5_Test, ones(268,1));

prob_test_1=exp(lp);               

%Realizamos el mismo proceso para cada uno de los train de Fold1                 
hyp_2 = minimize(hyp, @gp, -40, @infVB, meanfunc, covfunc, likfunc, Fold5_Train_2, Fold5_Train_2_Labels);
hyp_3 = minimize(hyp, @gp, -40, @infVB, meanfunc, covfunc, likfunc, Fold5_Train_3, Fold5_Train_3_Labels);
hyp_4 = minimize(hyp, @gp, -40, @infVB, meanfunc, covfunc, likfunc, Fold5_Train_4, Fold5_Train_4_Labels);
%hyp_2 = [];
%hyp_3 = [];
%hyp_4 = [];

[a b c d lp] = gp(hyp_2, @infVB, meanfunc, covfunc, likfunc, Fold5_Train_2, Fold5_Train_2_Labels, Fold5_Test, ones(268,1));
prob_test_2=exp(lp);  

[a b c d lp] = gp(hyp_3, @infVB, meanfunc, covfunc, likfunc, Fold5_Train_3, Fold5_Train_3_Labels, Fold5_Test, ones(268,1));
prob_test_3=exp(lp);

[a b c d lp] = gp(hyp_4, @infVB, meanfunc, covfunc, likfunc, Fold5_Train_4, Fold5_Train_4_Labels, Fold5_Test, ones(268,1));
prob_test_4=exp(lp);

%Calculamos la probabilidad media de pertenecer a la clase 1 y las curvas ROC y
%precisión respectivamente
prob_mean_Fold5 = (prob_test_1 + prob_test_2 + prob_test_3 + prob_test_4)/4;

[X1, Y1, T, AUC1] = perfcurve(Fold5_Test_Labels, prob_mean_Fold5, 1);

[X2, Y2, T2pr, AUC2] = perfcurve(Fold5_Test_Labels, prob_mean_Fold5, 1, 'xCrit', 'sens', 'yCrit', 'prec'); 

figure, plot(X1,Y1), xlabel('False positive rate'), ylabel('True positive rate'), title('ROC for Classification by Logistic Regression Fold5')
figure, plot(X2, Y2), xlabel('Sensitive rate'), ylabel('Precision rate'), title('ROC for Classification by Logistic Regression Fold5')

AUC1
AUC2

%Hacemos las predicciones fijando una threshold ?=0.5
prediciones = ones(268, 1);
prediciones(prob_mean_Fold5 <= 0.5) = -1;

%Calculamos matriz de confusión y métricas de bondad del clasificador
Confusion_Fold5 = confusionmat(Fold5_Test_Labels, prediciones, 'Order',[1 -1])
accuracy_Fold5 = sum(diag(Confusion_Fold5)) / sum(sum(Confusion_Fold5))
specificity_Fold5 = Confusion_Fold5(2,2) / sum(Confusion_Fold5(2,:))
sensitivity_Fold5 = Confusion_Fold5(1,1) / sum(Confusion_Fold5(1,:))
precision_Fold5 = Confusion_Fold5(1,1) / sum(Confusion_Fold5(:,1))
F_score_Fold5 = (2*precision_Fold5 * sensitivity_Fold5) / (precision_Fold5 + sensitivity_Fold5)
    