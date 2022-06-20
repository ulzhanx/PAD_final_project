# źródło danych [https://www.kaggle.com/c/titanic/](https://www.kaggle.com/c/titanic)

import streamlit as st
import pickle
from datetime import datetime
startTime = datetime.now()
# import znanych nam bibliotek

filename = "model_mashroom.sv"
model = pickle.load(open(filename,'rb'))
# otwieramy wcześniej wytrenowany model

cap_shape_d = {0:"bell",1:"conical",2:"convex", 3:"flat",4:"knobbed" ,5:"sunken" }
cap_surface_d = {0:"fibrous",1:"grooves", 2:"scaly", 3:"smooth" }
cap_color_d = {0:"brown",1:"buff",2:"convex", 3:"cinnamon",4:"gray" ,5:"green" ,
             6:"pink",7:"purple",8:"red", 9:"white",10:"yellow"}
bruises_d = {0:"bruises",1:"no" }
odor_d  = {0:"almond",1:"anise",2:"creosote", 3:"fishy",4:"musty" ,5:"none",6:"pungent",7:"spicy" }
gill_colo_d = {0:"attached",1:"descending",2:"free", 3:"notched"}

# o ile wcześniej kodowaliśmy nasze zmienne, to teraz wprowadzamy etykiety z ich nazewnictwem

def main():

	st.set_page_config(page_title="Mushroom Classification App")
	overview = st.container()
	left, right = st.columns(2)
	prediction = st.container()

	st.image("https://images.app.goo.gl/22tMgMXx1ayka2aZA")

	with overview:
		st.title("Mushroom Classification App")

	with left:

                cap_shape_radio = st.radio( "cap_shape", list(cap_shape_d.keys()), format_func=lambda x : cap_shape_d[x] )
                cap_surface_radio = st.radio( "cap_surface", list(cap_surface_d.keys()), format_func=lambda x : cap_surface_d[x] )
                cap_color_radio =st.radio( "cap_color", list(cap_color_d.keys()), format_func=lambda x : cap_color_d[x] )
                bruises_radio = st.radio( "bruises", list(bruises_d.keys()), format_func=lambda x : bruises_d[x] )
                odor_radio  = st.radio( "odor", list(odor_d.keys()), format_func=lambda x : odor_d[x] )
                gill_colo_radio = st.radio( "gill_colo", list(gill_colo_d.keys()), format_func=lambda x : gill_colo_d[x] )
	#with right:
		#age_slider = st.slider("Age", value=1, min_value=1, max_value=80)
		#sibsp_slider = st.slider("Siblings", min_value=0, max_value=10)
		#parch_slider = st.slider("Childrens", min_value=0, max_value=10)
		#fare_slider = st.slider("Fare", min_value=0, max_value=500, step=1)*/

	data = [[cap_shape_radio,
                cap_surface_radio, 
                cap_color_radio ,
                bruises_radio, odor_radio ,
                gill_colo_radio]]
	survival = model.predict(data)
	s_confidence = model.predict_proba(data)

	with prediction:
		st.subheader("edible or poisonous?")
		st.subheader(("edible" if survival[0] == 1 else "poisonous"))
		st.write("Prob {0:.2f} %".format(s_confidence[0][survival][0] * 100))

if __name__ == "__main__":
    main()

