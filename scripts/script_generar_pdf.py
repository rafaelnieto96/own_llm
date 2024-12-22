from reportlab.lib.pagesizes import LETTER
from reportlab.pdfgen import canvas
import os

def generar_resolucion_pdf(nombre_archivo):
    """Genera un PDF con una resolución jurídica de un juicio."""
    # Crear una carpeta de salida si no existe
    carpeta_salida = "salida"
    if not os.path.exists(carpeta_salida):
        os.makedirs(carpeta_salida)

    ruta_pdf = os.path.join(carpeta_salida, nombre_archivo)

    # Configurar el PDF
    c = canvas.Canvas(ruta_pdf, pagesize=LETTER)
    ancho, alto = LETTER

    # Título
    c.setFont("Times-Bold", 16)
    c.drawString(50, alto - 50, "RESOLUCIÓN JURÍDICA DEL JUICIO")

    # Subtítulo
    c.setFont("Times-Bold", 12)
    c.drawString(50, alto - 80, "Expediente: 12345-2024")
    c.drawString(50, alto - 100, "Juzgado: Sala Primera de lo Civil")

    # Contenido extenso de la resolución
    c.setFont("Times-Roman", 10)
    texto_resolucion = (
        "En la ciudad de Madrid, a los 22 días del mes de diciembre del año 2024, en una mañana fría y lluviosa, el juez Juan Manuel Torres García, "
        "representando a la Sala Primera de lo Civil, decide dar inicio a la presente resolución. Tras varias jornadas de deliberaciones, "
        "se ha llegado a una serie de conclusiones en torno al expediente número 12345-2024, iniciado por el demandante Roberto Sánchez Pérez "
        "contra la demandada Laura Gómez Hernández. Las argumentaciones presentadas abarcan múltiples áreas del derecho civil y contractual, "
        "reflejando un caso con notable complejidad y un alto grado de controversia entre las partes implicadas.\n\n"
        "El caso encuentra su origen en un contrato de compraventa firmado entre las partes el 10 de enero de 2024, mediante el cual la demandada se comprometió "
        "a entregar un vehículo marca Toyota, modelo Corolla del año 2020, a cambio de un pago de 15,000 euros. En este sentido, se destaca "
        "que el contrato incluyó cláusulas específicas relacionadas con el estado del vehículo, el cual debía estar en condiciones mecánicas "
        "óptimas, acompañado de la documentación requerida para su registro y circulación. Sin embargo, se presentó una discrepancia significativa "
        "cuando el demandante, tras realizar el pago el día 12 de enero de 2024, recibió un vehículo con fallos graves en el sistema de motor y con "
        "ausencia de ciertos documentos esenciales, como el certificado de inspección técnica y la tarjeta de propiedad actualizada.\n\n"
        "A lo largo del proceso judicial, el demandante presentó múltiples pruebas que incluyeron correos electrónicos, mensajes de texto, y un informe técnico "
        "elaborado por el taller AutoPro Madrid, en el cual se detallaban los daños mecánicos del vehículo. Este informe fue especialmente relevante "
        "para determinar que los problemas del motor no podían haber surgido tras la entrega, sino que eran preexistentes. Además, se presentó un testimonio "
        "de un experto mecánico, el cual sostuvo bajo juramento que el estado del vehículo era incompatible con las declaraciones realizadas por la demandada "
        "durante las negociaciones previas al contrato.\n\n"
        "Por su parte, la demandada argumentó que el demandante había aceptado el estado del vehículo al momento de su entrega y que la falta de documentos "
        "se debía a demoras administrativas ajenas a su control. No obstante, dichos argumentos no fueron respaldados con pruebas suficientes, lo que "
        "llevó a este juzgado a valorar de manera prioritaria los elementos probatorios presentados por el demandante.\n\n"
        "En cuanto a las consideraciones jurídicas, este juzgado ha analizado el caso bajo la luz del Código Civil, en particular los artículos 1254, 1124 y 1290. "
        "El artículo 1254 establece que el contrato existe desde que una o varias partes consienten en obligarse, lo que fue claramente demostrado en este caso. "
        "El artículo 1124 permite a la parte perjudicada por un incumplimiento contractual optar entre exigir el cumplimiento del contrato o su resolución, con "
        "resarcimiento de daños en ambos casos. Asimismo, el artículo 1290 aborda las medidas necesarias para garantizar el cumplimiento de las obligaciones.\n\n"
        "Por lo tanto, este juzgado resuelve lo siguiente:\n"
        "1. Declarar procedente la demanda interpuesta por Roberto Sánchez Pérez.\n"
        "2. Ordenar a Laura Gómez Hernández pagar una indemnización de 5,000 euros al demandante por los daños y perjuicios ocasionados, "
        "así como cubrir los costos del proceso judicial, los cuales ascienden a 2,500 euros adicionales.\n"
        "3. Exigir a la demandada que entregue los documentos faltantes en un plazo máximo de 15 días hábiles, bajo apercibimiento de "
        "imponer sanciones adicionales conforme a la ley.\n"
        "4. En caso de incumplimiento, habilitar al demandante para tomar medidas compensatorias inmediatas, incluyendo el embargo de bienes "
        "por un valor equivalente al monto adeudado.\n\n"
        "Finalmente, esta resolución se emite en plena conformidad con las disposiciones legales aplicables y quedará registrada en los archivos "
        "del juzgado. Se notifica a ambas partes que tienen un plazo de 10 días hábiles para interponer recurso de apelación, si así lo consideran necesario.\n\n"
        "Firmado: Juan Manuel Torres García, Juez de la Sala Primera de lo Civil."
    )



    # Dividir el texto en párrafos y páginas
    lineas = texto_resolucion.split("\n")
    x_inicial = 50
    y = alto - 130
    for linea in lineas:
        if y < 50:  # Salto de página si el espacio no es suficiente
            c.showPage()
            c.setFont("Times-Roman", 10)
            y = alto - 50
        c.drawString(x_inicial, y, linea)
        y -= 12

    # Guardar el PDF
    c.save()
    print(f"PDF generado en: {ruta_pdf}")

# Crear el PDF
nombre_archivo = "resolucion_juridica.pdf"
generar_resolucion_pdf(nombre_archivo)
