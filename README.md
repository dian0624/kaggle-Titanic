# kaggle-Titanic

此專案使用**jupyter notebook**製作，運用**numpy、pands、matplotlib、sklearn...**等模塊進行分析製圖，使用**隨機森林模型預測，獲得kaggle排名Top8%**。

--------------------------------------------------------------------------------------------------------
發現:
1. 透過卡方檢定，發現不同**艙等、性別、是否為15歲小孩、甲板、家庭型態、是否單獨、稱謂對是否生存「有相關」，設為重要特徵**。
2. 透過卡方檢定，發現不同**父母與小孩總數的群組對是否生存「沒有相關」**，其中不同**兄弟姊妹和配偶總數的群組對是否生存有相關，但提交後，分數下降歸類為型一錯誤**。
3. 透過F檢定，發現**不同艙等對年齡有顯著差異**，艙等可做年齡填充遺漏值依據。
4. 鐵達尼背景與結果呼應船長宣布婦女兒童優先離開，**年齡和性別為重要特徵**。
5. **船艙與甲板位置與逃生路徑有關**，字母越靠前越靠近輪船上層，艙等越好票價越高生存率也越高。
6. **港口生存率最高是由C港口上船，大多為一等艙客人**，而**生存率最低的S港口，3等艙客人佔了一半**，推測可能與當地經濟發展狀況有關。
7. 增加新特徵**有無同伴**，由親屬人數生存率可發現，當親屬人數大於 2 人時生存率會降低，考慮到船長宣言，優先逃難族群離開後，剩餘人員只能等待救援生存率隨之下降，以親屬劃分噪聲較大因此將範圍縮小為有無同伴後，有同伴者生存率高達五成。
8. 增加新特徵**稱謂**更加細分處理名字欄位，發現與船長宣言呼應，女士生存率高達7成、小男孩5成。

--------------------------------------------------------------------------------------------------------

## 過程
 一. 數據總覽
    1.train 損失值 : Age、Cabin、Embarked。
    2.test  損失值 : Age、Cabin、Fare。
    3. Cabin損失值最多。
    
 二.探索式分析 EDA
  
   1. 描述性統計
    
         ![image](https://github.com/dian0624/kaggle-Titanic/blob/master/Titanic_image/1587622580602.jpg)
            
   2. 相關矩陣
    
         ![image](https://github.com/dian0624/kaggle-Titanic/blob/master/Titanic_image/1587622809975.jpg)
            
         ![image](https://github.com/dian0624/kaggle-Titanic/blob/master/Titanic_image/1587622827232.jpg)
            
   3. 生存率與死亡率
    
      死亡 : 61.62%，總生存 : 38.38%。
      
         ![image](https://github.com/dian0624/kaggle-Titanic/blob/master/Titanic_image/1587622278163.jpg)
            
      類別型特徵:
      
         ![image](https://github.com/dian0624/kaggle-Titanic/blob/master/Titanic_image/1587623104207.jpg)
            
          1. Embarked港口: 從s港口登陸的人生存率最低，c港口生存率遠高於其他港口上船的人。 
            
          2. Pclass 艙等: 一等艙為最高等艙，生存率最高，越低等的艙等生存率有下降的趨勢。
            
          3.Sex 性別 1:male, 0:female :女性生存率遠大於男性，可能和當時船長宣布婦女兒童優先上救生艇有關。
            
          4.Parch 父母與小孩:同家族的父母和小孩的個數，獨自一人組存活率小於１～３人的群組，大於３個人的群組生存率反而更低，推測可能原因為群組會聽從船長指示優先讓媽媽和小孩離開，剩餘人員等待救援。
            
          5.SibSp 兄弟姊妹+配偶:兄弟姊妹和配偶的數目，獨自一人的存活率小於１～２個人的群組，大於３個人的群組生存率很低。
            
          6.根據Parch、SibSp，推斷有親屬的人生存率較高，但親屬3人以上生存率降低。
            
   4. Fare 票價 
      
        Fare票價非常態分佈，票價越高生存率有越高的趨勢，票價越高可能代表艙位等級越高，艙等可能也會有此趨勢。
          
        ![image](https://github.com/dian0624/kaggle-Titanic/blob/master/Titanic_image/1587623350406.jpg)
          
        ![image](https://github.com/dian0624/kaggle-Titanic/blob/master/Titanic_image/1587623416882.jpg)
          
   5. Age 年齡
      
        年齡非常態分佈但接近，0-15歲小孩存活率較高，5歲以下越明顯。
          
        ![image](https://github.com/dian0624/kaggle-Titanic/blob/master/Titanic_image/1587623575410.jpg)
          
        ![image](https://github.com/dian0624/kaggle-Titanic/blob/master/Titanic_image/1587623584448.jpg)
 
 三.特徵工程
 
   1. Embarked港口:以眾數填補遺漏值
   
   2. Pclass 艙等:
   
      1.**以95%信賴區間做卡方檢定，假設H0 不同艙等對於是否生存沒有相關，結果: p值 < 0.05，拒絕H0，不同的艙等對於是否生存有相關，艙等列為重要特徵**。
      
   3. Sex 性別 1:male, 0:female:
   
      1.**以95%信賴區間做卡方檢定，假設H0 不同性別對於是否生存沒有相關，結果: p值 < 0.05 ，拒絕H0，不同性別對於是否生存有相關，性別列為重要特徵**。
          
      2.做 LabelEncoder()編碼。
      
   4. Age 年齡:
   
      1.查看相關矩陣，發現**年齡和艙等有中度相關，推測年齡越大經濟能力越好，艙等越高，以95%信賴區間做做F檢定，H0 假設 不同艙等對年齡沒有顯著差異，結果: p值 < 0.05 ，拒絕H0，不同艙等對年齡有顯著差異，艙等可做年齡填充遺漏值依據。**
              
      2.**從Name中可以提取出稱謂**，發現有一個遺漏值，使用類似資料Title=Ms的年齡中位數填充。
      
      3.**新增特徵「Ischild」是否為小孩，以95%信賴區間做卡方檢定，H0 是否為小孩對是否生存沒有相關，結果: p值<0.05，拒絕H0，是否為15歲以上小孩對是否生存有相關**。
      
      4.**做boxcox，讓整體更趨近於常態**。
          
         ![image](https://github.com/dian0624/kaggle-Titanic/blob/master/Titanic_image/1587624939734.jpg)
      
       5.年齡分組，使用pandas.qcut切分為8組，**第一組生存最高，其中(16.23,18.758]相較其他組生存率較高、(13.778,14.233]生存率最低**，並做LabelEncoder()編碼。
      
         ![image](https://github.com/dian0624/kaggle-Titanic/blob/master/Titanic_image/1587625262987.jpg)
         
   5. Fare 票價:
      
        1.查看相關矩陣，發現**票價相關性，艙等、登船港口、父母小孩總數相關性較強，以此為基準填充遺漏值。**
          
        2.分組使用pandas.qcut切分為13組，**票價增高生存率也有增加的趨勢，尤其是最後一組(票價最高)生存率遠高於其他**，並做LabelEncoder()編碼。
          
         ![image](https://github.com/dian0624/kaggle-Titanic/blob/master/Titanic_image/1587625604200.jpg)
            
   6. Cabin 船艙:
      
       1.提取字母作為艙位名稱，並將遺漏值使用xx取代，進而觀察艙位分佈與艙等生存率關係，並做 LabelEncoder()編碼。
          
         ![image](https://github.com/dian0624/kaggle-Titanic/blob/master/Titanic_image/1587626134603.jpg)
         
             1.缺失值大部分都在3等艙，且上面已驗證[艙等與存活率有顯著差異]，從圖表可觀察到越低等的艙存活率越低。
            
             2.用艙等分類，根據船艙位置圖可發現與逃生路徑有關，字母越靠前越靠近輪船上層，A,B,C,T 和xx在pclass1最為相似歸類為一類，pclass2歸類一類，pclass3和G歸類一類，共3類。
            
   7. 新增特徵「Deck」甲板，以95%信賴區間做卡方檢定，H0 不同甲板對是否生存沒有相關，結果: p值 < 0.05，拒絕H0，**不同甲板對是否生存有相關，列為重要特徵。**
   
        ![image](https://github.com/dian0624/kaggle-Titanic/blob/master/Titanic_image/1587626309178.jpg)
          
   8. Parch 父母與小孩:
      
   1. 以95%信賴區間做卡方檢定，H0 不同父母小孩個數的群組對是否生存沒有相關，結果: p值 >0.05，接受H0，**不同父母小孩總數的群組對是否生存沒有相關**。
        
   9. SibSp 兄弟姊妹+配偶:
      
       1. 以95%信賴區間做卡方假設檢定，H0 不同兄弟姊妹和配偶個數的群組對是否生存沒有相關，結果: p值 <0.05，拒絕H0，**不同兄弟姊妹和配偶個數的群組對是否生存有相關，歸類為型一錯誤(提交後，分數下降)。**
        
  10. 將特徵Parch+SibSp形成**新特徵「Family」家庭成員**:
           
        ![image](https://github.com/dian0624/kaggle-Titanic/blob/master/Titanic_image/1587626638858.jpg)
               
              家庭成員介於1-3之間的群組生存率較高，其次為6、獨自一人，4-5生存率倒數第二，7-8生存率為0。
        
  11. 依據Family與生存率圖，新增特徵「Family_st」家庭型態， 獨自一人 -> 0， 小家庭 -> 1-3， 中家庭 -> 4-6， 大家庭 -> 7-10。
      
        1.以95%信賴區間做卡方檢定，H0 不同家庭型態對是否生存沒有相關，**結果: p值 < 0.05，拒絕H0，不同家庭型態對是否生存有相關**。
        
        ![image](https://github.com/dian0624/kaggle-Titanic/blob/master/Titanic_image/1587626993976.jpg)
            
            發現小家庭生存率最高，獨自一人第二，中家庭第三，大家庭0%生存率
        
  12. 根據Family、Family_st，推論「Alone」是否有同伴能夠減少噪聲，故新增特徵。
      
         1.以95%信賴區間做卡方檢定，H0 是否有同伴對是否生存沒有相關，結果: p值 < 0.05，拒絕Ho，**是否有同伴對是否生存有相關，列為特徵**，並做 LabelEncoder()編碼。
         
         ![image](https://github.com/dian0624/kaggle-Titanic/blob/master/Titanic_image/1587627307794.jpg)
         
  13. Title進一步處理分類，最後分為**男士、女士、小男孩以及特殊人員4組**。
      
       1.觀察4組生存率，發現Miss/Mrs/Ms組為女士，生存率最高，其次為Master，與船長宣言相呼應，女人與小孩先走。，以95%信賴區間做假設檢定，H0 不同稱謂對是否生存沒有顯著差異，結果: p值< 0.05，拒絕H0，**不同稱謂對是否生存有相關。**
        
        ![image](https://github.com/dian0624/kaggle-Titanic/blob/master/Titanic_image/1587627707711.jpg)
        
  14. 觀察整筆資料相關性矩陣，發現是否生存對 Family、Age、Embarked、Parch、SibSp、Family_st、IsChild 相關性較低，可以注意這幾個特徵。
      
        ![image](https://github.com/dian0624/kaggle-Titanic/blob/master/Titanic_image/1587627839091.jpg)
        
 四.特徵選擇
  
   1. 將特徵'Embarked'、 "Sex"、'Pclass'、"Family_st"、'Deck'、'Title'、'IsChild'、"Alone"使用獨熱編碼，並刪除用不到的特徵，'Name'、'Ticket'、'Family'、'Parch'、'SibSp'。
      
 五.模型選擇與預測
  
   1. 將data拆分為訓練集以及測試集，並訓練集拆分訓練集與驗證集，透過**AIC分數選擇模型，訓練模型和做交叉分析並查看訓練集與驗證集「準確率」，最後決定使用隨機森林模型**。  
      
   ![image](https://github.com/dian0624/kaggle-Titanic/blob/master/Titanic_image/1587628405285.jpg)
     
 六.分數提交Kaggle，**獲得排名前8%**。
  
  ![image](https://github.com/dian0624/kaggle-Titanic/blob/master/Titanic_image/1587621828514.jpg)
  
  
