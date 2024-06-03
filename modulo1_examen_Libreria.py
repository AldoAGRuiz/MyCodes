import pandas as pd
import seaborn as sns
import base64
from io import BytesIO
from IPython.display import HTML
import numpy as np
from datetime import date
from itertools import combinations
from scipy import integrate
from scipy.stats import zscore
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression, f_classif
from sklearn.feature_selection import chi2, mutual_info_classif, mutual_info_regression
from varclushi import VarClusHi



class examen:
	
    def __init__(self):
        pass


    def clasificar_variables_numericas(self,df):
        if not isinstance(df, pd.DataFrame):
            return "El argumento proporcionado no es un DataFrame."
        if df.empty:
            return "El DataFrame está vacío."

        info_variables = []
        discretas = []
        continuas = []

        for columna in df.columns:
            if df[columna].dtype in ['int64', 'float64']:
                num_unicos = len(df[columna].unique())
                tipo = "Discreta" if num_unicos <= 15 else "Continua"
                info_variables.append({'Columna': columna, 'Num_Unicos': num_unicos, 'Tipo': tipo})
                if num_unicos <= 15:
                    discretas.append(columna)
                else:
                    continuas.append(columna)

        df_info = pd.DataFrame(info_variables)
        return df_info, discretas, continuas

    def clasificar_variables_nonumericas(self,df):
        if not isinstance(df, pd.DataFrame):
            return "El argumento proporcionado no es un DataFrame."
        if df.empty:
            return "El DataFrame está vacío."

        info_variables = []
        discretas = []
        continuas = []

        for columna in df.columns:
            if df[columna].dtype not in ['int64', 'float64','int32','int16','int','int8','float32','float','float16','float8']:
                num_unicos = len(df[columna].unique())
                tipo = "Discreta" if num_unicos <= 15 else "Continua"
                info_variables.append({'Columna': columna, 'Num_Unicos': num_unicos, 'Tipo': tipo})
                if num_unicos <= 15:
                    discretas.append(columna)
                else:
                    continuas.append(columna)

        df_info = pd.DataFrame(info_variables)
        return df_info, discretas, continuas


    def chequear_completitud(self,df):
        if not isinstance(df, pd.DataFrame) or df.empty:
            return "El objeto de entrada no es un DataFrame o está vacío."

        total_filas = df.shape[0]
        faltantes_por_columna = df.isna().sum()
        filas_restantes = df.dropna().shape[0]
        filas_borradas = total_filas - filas_restantes

        # Calcular el porcentaje de completitud
        completitud = 100 * (1 - faltantes_por_columna / total_filas)

        resultados = pd.DataFrame({
            'Columna': faltantes_por_columna.index,
            'Valores Faltantes': faltantes_por_columna.values,
            'Completitud (%)': completitud.values
        })

        # Agregar filas resumen al final
        resultados_totales = pd.DataFrame({
            'Columna': ['Total na en el dataframe', 'Filas Restantes si se borran los NAs', 'Filas que se eliminarian si existe al menos un na', 'Tamaño Original del DataFrame'],
            'Valores Faltantes': [faltantes_por_columna.sum(), filas_restantes, filas_borradas, total_filas],
            'Completitud (%)': [None, None, None, None]  # No aplica completitud para las filas resumen
        })

        resultados = pd.concat([resultados, resultados_totales], ignore_index=True)
        return resultados


    def graficar_histogramas(self,df, columnas, num_col):  # num_col con valor predeterminado 4
        if not isinstance(df, pd.DataFrame) or df.empty:
            print("El argumento proporcionado no es un DataFrame de Pandas o está vacío")
            return
        num_col = num_col+1
        # Ajustar el porcentaje de ancho basado en num_col
        width_percent = 100 // num_col
        
        # Iniciar el HTML
        html_str = '<div style="display:flex;flex-wrap:wrap;">'

        for columna in columnas:
            # Verificar si la columna existe en el DataFrame
            if columna not in df.columns:
                print(f"La columna '{columna}' no existe en el DataFrame.")
                continue

            # Crear la gráfica de histograma
            ax = sns.histplot(df[columna])
            buffer = BytesIO()
            ax.figure.savefig(buffer, format='png')
            buffer.seek(0)
            # Codificar la imagen en base64 y decodificarla para incrustarla en HTML
            image_png = buffer.getvalue()
            buffer.close()
            graphic = base64.b64encode(image_png)
            graphic = graphic.decode('utf-8')
            # Añadir la imagen al HTML. El ancho se ajusta dinámicamente según num_col
            html_str += f'<div style="width:{width_percent}%;margin:0.5%;"><img src="data:image/png;base64,{graphic}" style="width:100%;"/></div>'

            ax.figure.clf()

        # Cerrar el div del HTML
        html_str += '</div>'
        display(HTML(html_str))

    def mapa_calor(self,datos):
        if isinstance(datos, pd.DataFrame):
            datos = datos.select_dtypes(include=['int', 'float'])
            matriz_correlacion = datos.corr()

            # Ajustar el tamaño general de los elementos del gráfico
            sns.set(rc={'figure.figsize':(13,8)})

            # Dibujar el mapa de calor directamente
            sns.heatmap(matriz_correlacion, annot=True, annot_kws={'size': 6})

            # Restablecer los cambios realizados por sns.set
            sns.reset_orig()
        else:
            print("Debes ingresar un DataFrame.")


    def graficar_bh(self,df, columnas, num_col):
        if not isinstance(df, pd.DataFrame) or df.empty:
            print("El argumento proporcionado no es un DataFrame de Pandas o está vacío")
            return
        num_col = num_col +1
        width_percent = 100 // num_col  # Se convierte a entero para simplificar el cálculo
        html_str = '<div style="display:flex;flex-wrap:wrap;">'

        for columna in columnas:
            if columna not in df.columns:
                print(f"La columna '{columna}' no existe en el DataFrame.")
                continue

            # Crear la gráfica de barras horizontales usando value_counts para ambos, x e y
            conteos = df[columna].value_counts()
            ax = sns.barplot(x=conteos, y=conteos.index, orient="h")  # x contiene los conteos, y los valores únicos correspondientes

            # Guardar la gráfica en un buffer
            buffer = BytesIO()
            ax.figure.savefig(buffer, format='png')
            buffer.seek(0)

            # Codificar la imagen en base64 y decodificarla para incrustarla en HTML
            image_png = buffer.getvalue()
            buffer.close()
            graphic = base64.b64encode(image_png)
            graphic = graphic.decode('utf-8')

            # Añadir la imagen al HTML. El ancho se ajusta dinámicamente según num_col
            html_str += f'<div style="width:{width_percent}%;margin:0.5%;"><img src="data:image/png;base64,{graphic}" style="width:100%;"/></div>'

            # Limpiar la figura después de guardarla
            ax.figure.clf()

        # Cerrar el div del HTML y mostrar los resultados
        html_str += '</div>'
        display(HTML(html_str))





    def graficar_densidades(self,df,columnas,num_col):
        if not isinstance(df, pd.DataFrame):
            return "El argumento proporcionado no es un DataFrame."
        if df.empty:
            return "El DataFrame está vacío."
        
         
        # Establecer el número de columnas para los gráficos
        num_col = num_col +1
        html_str = '<div style="display:flex;flex-wrap:wrap;">'
        
        # Calcular el ancho de cada imagen basado en el número de columnas deseado
        width = 100 // num_col
        
        for cont in columnas:
            ax = sns.kdeplot(df[cont], label=cont, fill=True)
            ax.set_title(f'Gráfico de Densidad para {cont}')
            ax.set_xlabel('Valor')
            ax.set_ylabel('Densidad')
            
            # Guardar la gráfica en un buffer
            buffer = BytesIO()
            ax.figure.savefig(buffer, format='png')
            buffer.seek(0)
            
            # Codificar la imagen en base64 y decodificarla para incrustarla en HTML
            image_png = buffer.getvalue()
            buffer.close()
            graphic = base64.b64encode(image_png)
            graphic = graphic.decode('utf-8')
            
            # Añadir la imagen al HTML. El ancho se ajusta dinámicamente según 'num_col'
            html_str += f'<div style="width:{width}%;margin:0.5%;"><img src="data:image/png;base64,{graphic}" style="width:100%;"/></div>'
            
            # Limpiar la figura actual para la siguiente gráfica
            ax.figure.clf()

        # Cerrar el div del HTML y mostrar los resultados
        html_str += '</div>'
        display(HTML(html_str))




    def iqr(self,df,varc):
        var= [col for col in df.columns if (df[col].dtype == 'float64') or (df[col].dtype == 'int64')]
        df_h=df[var]
        Q1 = df_h.quantile(0.25)
        Q3 = df_h.quantile(0.75)
        IQR = Q3 - Q1 
        mask_q =  ~(((df_h < (Q1 - 1.5 * IQR)) | (df_h > (Q3 + 1.5 * IQR)))).any(axis=1)
        print(f'Tamaño original {df_h.shape[0]}', f'Tras aplicar IQR: {df_h[mask_q].shape[0]} registros')
        df_h=df_h[mask_q]
        return(df_h)
    def z_score(self,df,varc):
        var= [col for col in df.columns if (df[col].dtype == 'float64') or (df[col].dtype == 'int64')]
        df_h=df[var]
        Z_scores = df_h.apply(zscore)
        mask_z = (Z_scores > -3) & (Z_scores < 3)
        print(f'Tamaño original: {df_h.shape[0]}', f'Al aplicar el método Z-Score: {df_h[mask_z].shape[0]}')
        df_h=df_h[mask_z]
        return(df_h)
    def iso_forest(self,df,varc):
        var_num = [col for col in df.columns if (df[col].dtype == 'float64') or (df[col].dtype == 'int64')]
        df_h=df[varc]
        df_m=df[var_num]
        iso_forest = IsolationForest(n_estimators=100, random_state=42)
        iso_forest.fit(df_m)
        iso_preds = iso_forest.predict(df_m)
        df_iso=df[iso_preds==1]
        print(f'Tamaño original: {df_h.shape[0]}',f'Al aplicar el método Iso Forest: {df_iso.shape[0]}')
        return (df_iso)

    def encontrar_num_optimo_pca(self,X, umbral_varianza=0.9):
        # Estandarización de los datos
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        num_componentes = list(range(1, 6))  # Hasta el número total de características
        varianza_acumulada = 0
        for n in num_componentes:
            pca = PCA(n_components=n)
            pca.fit(X_scaled)  # Utilizamos los datos estandarizados
            varianza_acumulada = sum(pca.explained_variance_ratio_)
            if varianza_acumulada >= umbral_varianza:
                return n  # Retorna el número óptimo de componentes
        return len(num_componentes)  # En caso de que sea necesario usar todas las características
    
    def identificacion_clusters(self,df):
        vc = VarClusHi(df)
        grupos= vc.varclus() # Ejecutar el algoritmo de agrupación de VarClusHi
        conjuntos= vc.rsquare
        return(conjuntos)
    
    def identificacion_1v_cluster(self, df):
        vc = VarClusHi(df)
        vc.varclus()  # Ejecutar el algoritmo de agrupación de VarClusHi
        variables_seleccionadas = []
        for i in range(vc.rsquare['Cluster'].nunique()):
        # Filtrar las variables del cluster actual
            cluster_i = vc.rsquare[vc.rsquare['Cluster'] == i]
        # Ordenar las variables del cluster por 'RS_Ratio' y seleccionar la de menor valor
            mejor_variable = cluster_i.sort_values(by='RS_Ratio', ascending=True).iloc[0]['Variable']
            variables_seleccionadas.append(mejor_variable)
        # Crear un nuevo DataFrame solo con las variables seleccionadas
        df_reducido = df[variables_seleccionadas]
        return df_reducido
    
    def pca_funcion(self,df,k=2):
        # Paso 1: Estandarizar los datos
        scaler = StandardScaler()
        scaled_df = scaler.fit_transform(df)
        # Paso 2: Aplicar PCA
        pca = PCA(n_components=k)  # Nos interesan las dos primeras componentes principales
        principalComponents = pca.fit_transform(scaled_df)
        componente_df = pd.DataFrame(principalComponents)
        return(componente_df)
    
    def seleccionar_kbest(self,df, variable_objetivo, num_variables_deseadas=10, funcion=f_classif):
        X = df.drop(columns=[variable_objetivo])
        y = df[variable_objetivo]
        kb = SelectKBest(k=num_variables_deseadas, score_func=funcion)
        kb.fit(X,y)
        variables_seleccionadas = list(kb.get_feature_names_out())
        return variables_seleccionadas
    
    
    
    

    def WOE_IV(self,df, var, tgt):
        aux = df[[var, tgt]].groupby(var).agg(["count", "sum"])
        aux["evento"] = aux[tgt, "sum"]
        aux["no_evento"] = aux[tgt, "count"] - aux[tgt, "sum"]
        aux["%evento"] = aux["evento"] / aux["evento"].sum()
        aux["%no_evento"] = aux["no_evento"] / aux["no_evento"].sum()
        aux["WOE"] = np.log(aux["%no_evento"] / aux["%evento"])
        aux["IV"] = (aux["%no_evento"] - aux["%evento"])*aux["WOE"]
        return aux["IV"].sum()
    

    def discretizar_variables(self,df, X):
        for variable in X.columns:
            df[f"C_{variable}"] = pd.cut(df[variable], bins=10)
        return df



    def ranking_iv(self,X, df, poder_p,target,ls_discretized):
        for n_bins in range(2, poder_p):
            for var in X.columns:
                df[f"C_{var}"] = pd.qcut(df[var], q=n_bins, duplicates="drop").cat.add_categories(["Missing"]).fillna("Missing").astype(str)
            ls_discretized = [x for x in df.columns if x.startswith("C_")]
            df_iv = pd.DataFrame(columns=["iv"])
            
            for var in ls_discretized:
                df_iv.loc[var, "iv"] = self.WOE_IV(df = df, var = var, tgt = target)
            
        df_iv = pd.DataFrame(columns=["iv"])
        for var in ls_discretized[:50]:
            df_iv.loc[var, "iv"] = self.WOE_IV(df = df, var = var, tgt = "target")
        return df_iv.sort_values(by="iv", ascending=False)
        




