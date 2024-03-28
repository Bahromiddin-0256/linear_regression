import matplotlib.pyplot as plt
import numpy as np
import pandas  as pd
import matplotlib.pyplot as plt 
import seaborn as sns
from scipy.stats import boxcox
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression



def main():
    plt.rcParams['figure.figsize'] = [8,5]
    plt.rcParams['font.size'] =14
    plt.rcParams['font.weight']= 'bold'
    df = pd.read_csv('insurance.csv')
    print('\nUstunlar va qatorlar soni: ',df.shape)
    print(df.head())
    sns.lmplot(x='bmi',y='charges',data=df,aspect=2,height=6)
    plt.xlabel('Tana massasi indexsi $(kg/m^2)$: mustaqil o\'zgaruvchi (x) sifatida')
    plt.ylabel('Sug\'urta to\'lovlari: qaram o\'zgaruvchi (y) sifatida')
    plt.title('BMI va Sug\'urta to\'lovlari orasidagi moslik: \n');
    
    plt.savefig('linear_regression.png')
    
    f= plt.figure(figsize=(12,4))

    ax=f.add_subplot(121)
    sns.distplot(df['charges'],bins=50,color='r',ax=ax)
    ax.set_title('Sug\'urta to\'lovlarini taqsimlash')

    ax=f.add_subplot(122)
    sns.distplot(np.log10(df['charges']),bins=40,color='b',ax=ax)
    ax.set_title('Sug\'urta to\'lovlarini $log$ shkalasida taqsimlash')
    ax.set_xscale('log')
    f.savefig('charges_dist.png')
    
    categorical_columns = ['sex','children', 'smoker', 'region']
    df_encode = pd.get_dummies(data = df, prefix = 'OHE', prefix_sep='_',
                columns = categorical_columns,
                drop_first =True,
                dtype='int8')
    y_bc,lam, ci= boxcox(df_encode['charges'],alpha=0.05)
    df_encode['charges'] = np.log(df_encode['charges'])
    
    X = df_encode.drop('charges',axis=1)
    y = df_encode['charges']

    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=23)
    X_train_0 = np.c_[np.ones((X_train.shape[0],1)),X_train]
    X_test_0 = np.c_[np.ones((X_test.shape[0],1)),X_test]
    theta = np.matmul(np.linalg.inv( np.matmul(X_train_0.T,X_train_0) ), np.matmul(X_train_0.T,y_train))     
    parameter = ['theta_'+str(i) for i in range(X_train_0.shape[1])]
    columns = ['intersect:x_0=1'] + list(X.columns.values)
    parameter_df = pd.DataFrame({'Parameter':parameter,'Columns':columns,'theta':theta})
    lin_reg = LinearRegression()
    lin_reg.fit(X_train,y_train) # Note: x_0 =1 is no need to add, sklearn will take care of it.

    #Parameter
    sk_theta = [lin_reg.intercept_]+list(lin_reg.coef_)
    parameter_df = parameter_df.join(pd.Series(sk_theta, name='Sklearn_theta'))
    y_pred_sk = lin_reg.predict(X_test)
    print("X test", X_test) 
    print("Y prediction ", y_pred_sk)



if __name__ == "__main__":
    main()