import pandas as pd
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

docs=["hôm nay tôi đi học",
"ngày mai tôi ở nhà",
"hôm đi học tôi không đi chơi được",
"ngày mai ở nhà tôi sẽ đi chơi",
"nếu không đi chơi thì tôi đi ngủ"
]

# khởi rạo CountVectorizer()
cv=CountVectorizer()
# Đếm số lượng mỗi từ
word_count_vector=cv.fit_transform(docs)
print("word_count_vector")
print(word_count_vector.shape)

tfidf_transformer=TfidfTransformer(smooth_idf=True,use_idf=True)
tfidf_transformer.fit(word_count_vector)

df_idf = pd.DataFrame(tfidf_transformer.idf_, index=cv.get_feature_names(),columns=["idf_weights"])
print("df_idf của kho từ vựng:")
print(df_idf)
print("chữ thứ 2:")
print(df_idf.index[2])
print("giá trị df của chữ thứ 2:")
print(df_idf.values[2])
# sort ascending
print(df_idf.sort_values(by=['idf_weights']))

# tính toán ma trận
count_vector=cv.transform(docs)
print(count_vector.toarray())
# tính giá trị tf-idf
tf_idf_vector = tfidf_transformer.transform(count_vector)
print("vector tf_idf:")
print(tf_idf_vector[1])
feature_names = cv.get_feature_names()
# chọn vector tfidf tương ứng với mỗi văn bản
document_vector=tf_idf_vector[1]
# chuyển đổi để in ra
df = pd.DataFrame(document_vector.T.todense(), index=feature_names, columns=["tfidf"])
print("vector tfidf của văn bản đã chọn:")
print(df)
# có thể chọn sắp theo thứ tự - giống như pad trong deep learning
# print(df.sort_values(by=["tfidf"],ascending=False))

# settings that you use for count vectorizer will go here
tfidf_vectorizer=TfidfVectorizer(use_idf=True)
# just send in all your docs here
tfidf_vectorizer_vectors=tfidf_vectorizer.fit_transform(docs)
print("Sau khi fit_transform văn bản thứ 2:")
print(tfidf_vectorizer_vectors[1])
