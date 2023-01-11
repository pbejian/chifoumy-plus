#-------------------------------------------------------------------------------
import streamlit as st
import pickle
#from include import picture_to_df, espace
import include as inc
#-------------------------------------------------------------------------------

html_title = "<h2 style='color:#FF036A'>Chifoumy plus: pierre, feuille, ciseaux, python et Spock !</h2>"
st.markdown(html_title, unsafe_allow_html=True)

st.markdown("""
Cette version étendue a été popularisée par la série « The Big Bang Theory ». Voici
[une vidéo de Sheldon Cooper](https://youtu.be/_PUEoDYpUyQ)
expliquant les règles du jeux à cinq positions. Nous avons remplacé le lézard par un python... car nous programmons en Python !
""", unsafe_allow_html=True)


html_subtitle = "<h3 style='color:#44B7E3'>Testons la reconnaissance des cinq gestes.</h3>"
st.markdown(html_subtitle, unsafe_allow_html=True)

html_subtitle = "<p style='color:#000000'>NB - Pour une meilleure reconnaissance, approchez votre main de la camera .</p>"
st.markdown(html_subtitle, unsafe_allow_html=True)

picture = None
picture = st.camera_input(label=" ", disabled=False, key=666)
if picture:
    button1 = st.button("Tester la photo", key=1234)
    if button1:
        df = inc.picture_to_df(picture)
        # st.write(type(df))
        if type(df) == type("toto"):
            st.write("Problème dans l'acquisition photo.")
        else:
            # Loading the trained scaler and the trained model
            my_scaler_nospock = pickle.load(open("models/scaler_spock.pkl", "rb"))
            my_model_nospock = pickle.load(open("models/model_spock.pkl", "rb"))

            # Scaling the new dataframe
            X_new = df
            X_new_scaled = my_scaler_nospock.transform(X_new)

            # Prediction with the new data
            target = my_model_nospock.predict(X_new_scaled)
            target = target[0]
            html_pierre ="<h3 style='color:#44B7E3'>Votre geste : pierre</h3>"
            html_feuille ="<h3 style='color:#44B7E3'>Votre geste : feuille</h3>"
            html_ciseaux ="<h3 style='color:#44B7E3'>Votre geste : ciseaux</h3>"
            html_python ="<h3 style='color:#44B7E3'>Votre geste : python</h3>"
            html_spock ="<h3 style='color:#44B7E3'>Votre geste : Spock</h3>"
            chifoudict = {0: html_pierre, 1: html_feuille, 2: html_ciseaux,
                          3: html_python, 4: html_spock}
            html_gesture = chifoudict[target]
            st.markdown(html_gesture, unsafe_allow_html=True)

#-------------------------------------------------------------------------------
# Conclusion avec le lien vers les sources sur GitHub

inc.espace(2)
st.markdown("""
    <hr>
""", unsafe_allow_html=True)
inc.espace(2)
st.write("""
📝 Sources de l'application :
[https://github.com/pbejian/chifoumy-plus/](https://github.com/pbejian/chifoumy-plus/)
""")
#-------------------------------------------------------------------------------
