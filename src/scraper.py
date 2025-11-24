from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.alert import Alert
from selenium.common.exceptions import NoAlertPresentException
from typing import List, Optional
from datetime import datetime


import time
import numpy as np
import re
import pandas as pd
import os
import csv
import unicodedata




class Scraper:
    def __init__(self, headless: bool = False, implicit_wait: int = 5, timeout: int = 10):
        options = webdriver.ChromeOptions()
        if headless:
            options.add_argument("--headless=new")
        options.add_argument("--start-maximized")
        options.add_argument("--disable-extensions")
        # opciones para redirigir descargas automáticas de PDF
        self.download_dir = os.path.abspath('../download_pdf')
        # Crear directorio si no existe
        os.makedirs(self.download_dir, exist_ok=True)
        preferences = {
            "download.default_directory": self.download_dir,
            "download.prompt_for_download": False,
            "download.directory_upgrade": True,
            "safebrowsing.enabled": True
        }
        print(self.download_dir)
        options.add_experimental_option("prefs", preferences)
        service = Service(ChromeDriverManager().install())
        self.driver = webdriver.Chrome(service=service, options=options)
        self.wait = WebDriverWait(self.driver, timeout)
        self.driver.implicitly_wait(implicit_wait)
        self.SKIP_TITLES = {
            self._norm("Sendero accesible del Cornato en valle de Pineta"),
            self._norm("Sendero adaptado de la Pradera de Ordesa"),
            self._norm("Itinerario adaptado del puente de la Gorga"),
            self._norm("Itinerario adaptado al lago de Sant Maurici"),
            self._norm("Itinerario adaptado hasta el mirador del Cap del Ras"),
            self._norm("Itinerario adaptado hasta el mirador de Els Orris"),
        }
        print("Navegador iniciado")

    def start(self, url: str):
        self.driver.get(url)

    def quit(self):
        self.driver.quit()
        print("Navegador cerrado")

    def close_popup(self):
        try:
            btn = self.wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, "div.btn_cerrar")))
            btn.click()
            print("Pop-up cerrado")
        except Exception as e:
            print("No se encontró el pop-up:", e)

    def handle_subscription(self, subscribe: bool = True, name: str = "Nombre", email: str = "email@example.com", timeout: int = 10):
        """
        Gestiona el popup de suscripción basado en el HTML proporcionado.
        - Si subscribe=True rellena NombreSubscripcion + EmailSubscripcion, marca checkbox y clica EnviarSubscripcion.
        - Si subscribe=False clica 'No gracias...' para saltar la suscripción.
        Devuelve True si la acción se realizó o no había popup, False si falló.
        """
        w = WebDriverWait(self.driver, timeout)
        try:
            form = w.until(EC.visibility_of_element_located((By.ID, "form_subscripcion")))
            # elementos claros por ID
            try:
                name_input = form.find_element(By.ID, "NombreSubscripcion")
                email_input = form.find_element(By.ID, "EmailSubscripcion")
                checkbox = form.find_element(By.ID, "accepto_politica_input")
                send_div = form.find_element(By.ID, "EnviarSubscripcion")
            except Exception:
                print("No se localizaron todos los campos del formulario")
                return False
            if subscribe:
                name_input.clear()
                name_input.send_keys(name)
                email_input.clear()
                email_input.send_keys(email)
                if not checkbox.is_selected():
                    checkbox.click()
                send_div.click()
                time.sleep(10)
                try:
                    alert = Alert(self.driver)
                    print("Mensaje del alert:", alert.text)
                    alert.accept()  # Hace clic en "Aceptar"
                    print("Alert aceptado correctamente.")
                except NoAlertPresentException:
                    print("No se mostró ningún alert.")
                print("Formulario de suscripción enviado")
                time.sleep(10)
            else:
                no_gracias = form.find_element(By.XPATH, ".//div[contains(., 'No gracias') or contains(., 'ya recibo la newsletter')]")
                self._click(no_gracias)
                time.sleep(0.6)
                print("Se pulsó 'No gracias' para saltar la suscripción")
        except Exception:
            # no hay popup
            print("No se detectó el popup de suscripción")
            return True
                  
    def click_guides_menu(self):
        try:
            guia_web = self.wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, "#item_menu_top_78 a")))
            guia_web.click()
            print("Click en 'Guías Web y PDF gratuitas'")
        except Exception as e:
            print("No se pudo hacer clic en el menú:", e)

    # ---- helpers ----
    def _click(self, el):
        try:
            self.driver.execute_script("arguments[0].scrollIntoView({block:'center'});", el)
            time.sleep(0.15)
            self.driver.execute_script("arguments[0].click();", el)
        except Exception:
            try:
                el.click()
            except Exception as e:
                print("Fallo al clicar elemento:", e)

    def save_route_to_csv(self, route_data: dict, csv_path: str):
        """
        Guarda los datos de una ruta en un archivo CSV.
        - Crea la carpeta si no existe.
        - Crea el archivo si no existe (con cabecera).
        - Añade una nueva fila en cada llamada.
        """
        folder = os.path.dirname(csv_path)
        if folder:
            os.makedirs(folder, exist_ok=True)

        write_header = not os.path.exists(csv_path)

        try:
            with open(csv_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=route_data.keys())
                if write_header:
                    writer.writeheader()
                writer.writerow(route_data)
            print(f"Ruta guardada en CSV: {csv_path}")
        except Exception as e:
            print(f"Error al guardar CSV: {e}")

    def get_location_info(self) -> dict:
        location = {
            "pueblo": "",
            "zona": "",
            "comarca": "",
            "provincia": "",
            "region": "",
            "pais": ""
        }

        try:
            # Prueba con varios selectores (la web usa distintos layouts)
            candidates = [
                ".home_cadena_zonas",
                ".ruta_cadena",
                ".breadcrumbs",
                "#cadena_zonas"
            ]
            cadena = None
            for sel in candidates:
                try:
                    cadena = self.driver.find_element(By.CSS_SELECTOR, sel)
                    break
                except Exception:
                    continue

            if not cadena:
                raise Exception("No se encontró ningún bloque de cadena de zonas")

            links = cadena.find_elements(By.TAG_NAME, "a")
            for link in links:
                title = link.get_attribute("title")
                text = link.text.strip()
                if title and "Rutas en " in title:
                    if not location["pueblo"] and len(links) > 5:
                        location["pueblo"] = text
                    elif not location["zona"] and len(links) > 4:
                        location["zona"] = text
                    elif not location["comarca"] and len(links) > 3:
                        location["comarca"] = text
                    elif not location["provincia"] and len(links) > 2:
                        location["provincia"] = text
                    elif not location["region"] and len(links) > 1:
                        location["region"] = text
                    elif not location["pais"] and len(links) > 0:
                        location["pais"] = text

        except Exception as e:
            print(f"Error extrayendo información de localización: {e}")

        return location


    def get_route_details(self,route_title) -> dict:
        """
        Extrae todos los detalles de la ruta actual incluyendo localización
        """
        try:
            # Obtener localización
            location = self.get_location_info()
            
            # Obtener contenedor de detalles técnicos
            box = self.wait.until(EC.presence_of_element_located((By.CLASS_NAME, "ruta_box")))
            
            def get_text_by_span(container, span_text: str) -> str:
                try:
                    span = container.find_element(By.XPATH, f".//span[text()='{span_text}']")
                    return span.find_element(By.XPATH, "./following-sibling::p").text.strip()
                except Exception:
                    return ""
            # Obtener valor dificultad y la descripción 
            try:
                dif_box = self.driver.find_element(By.CSS_SELECTOR, "div.ruta_box_dif")
                dificultad = dif_box.find_element(By.CSS_SELECTOR, "p").text.strip()
                try:
                    descripcion_dificultad = self.driver.execute_script(
                        "return arguments[0].innerText;", 
                        dif_box.find_element(By.CSS_SELECTOR, "span.classic")
                    ).strip()
                except Exception:
                    descripcion_dificultad = ""
            except Exception:
                dificultad = ""
                descripcion_dificultad = ""

            # Extraer categorías
            categories = []
            try:
                types_div = self.driver.find_element(By.ID, "tipo_rutas_ficha")
                for a in types_div.find_elements(By.CSS_SELECTOR, "ul li a"):
                    if title := a.get_attribute("title"):
                        categories.append(title.strip())
            except Exception:
                pass

            # URL y ruta del PDF
            current_url = self.driver.current_url
            pdf_name = 'RUTAS-PIRINEOS-' + current_url.split('/')[-1] + '_es.pdf'
            pdf_path = os.path.join(self.download_dir, pdf_name)

            # Construir diccionario con toda la información
            route_data = {
                "Pais": location["pais"],
                "Region": location["region"],
                "Provincia": location["provincia"],
                "Comarca": location["comarca"],
                "Zona": location["zona"],
                "Pueblo": location["pueblo"],
                "Nombre de la ruta": route_title,
                "Categoria": ";".join(categories),
                "Dificultad": dificultad,
                "Descripcion dificultad": descripcion_dificultad,
                "Distancia Total": get_text_by_span(box, "Distancia total"),
                "Altitud minima": get_text_by_span(box, "Altitud mínima").replace("m", ""),
                "Altitud máxima": get_text_by_span(box, "Altitud máxima").replace("m", ""),
                "Desnivel acumulado": get_text_by_span(box, "Desnivel acumulado"),
                "Tiempo total efectivo": get_text_by_span(box, "Tiempo total efectivo"),
                "Punto de salida / llegada": get_text_by_span(box, "Punto de salida / llegada"),
                "Población más cercana": get_text_by_span(box, "Población más cercana"),
                "Link archivo": current_url,
                "Pdf_path": pdf_path,
                "Fecha": datetime.now().strftime("%d/%m/%Y")
            }
            
            return route_data
        except Exception as e:
            print(f"Error extrayendo detalles de la ruta: {e}")
            return {}
 
    def wait_for_pdf_button(self, timeout: Optional[int] = None):
        """
        Espera específicamente al enlace de descarga dentro de <div id="pdf_ruta">.
        Devuelve el elemento si está clickable o None si no aparece en el timeout.
        """
        
        w = self.wait if timeout is None else WebDriverWait(self.driver, timeout)
        
        selectors = [
            "#pdf_ruta_a",  # por ID
            "a.link_rojo[onclick*='Descargas_downloads']",  # por clase y atributo onclick
            "a[rel='alternate'][type='application/pdf']",  # por atributos específicos
            "a[title='Descargar ruta pdf']"  # por título
        ]
        
        for selector in selectors:
            try:
                pdf_button = w.until(EC.element_to_be_clickable((By.CSS_SELECTOR, selector)))
                print(f"Botón PDF encontrado con selector: {selector}")
                return pdf_button
            except Exception:
                continue
                
        print("No se encontró el botón PDF con ningún selector")
        return None
    

    def download_pdf(self, pdf_button, wait_after_click: float = 1.0):
        """
        Clic en el botón PDF y manejo de nuevas pestañas si se abren.
        Mantiene la ventana original como foco al finalizar.
        """
        if pdf_button is None:
            print("Botón PDF no disponible")
            return False
        try:
            original_windows = list(self.driver.window_handles)
            pdf_button.click()
            time.sleep(wait_after_click)
            new_windows = list(self.driver.window_handles)
            # si se abrió una nueva pestaña, cerrarla y volver a la original
            if len(new_windows) > len(original_windows):
                new_tab = [w for w in new_windows if w not in original_windows][0]
                self.driver.switch_to.window(new_tab)
                time.sleep(0.3)
                # si la nueva pestaña muestra el PDF, cerramos esa pestaña
                self.driver.close()
                self.driver.switch_to.window(original_windows[0])
                print("PDF abierto en nueva pestaña y cerrada")
            else:
                # descarga directa o apertura en la misma ventana
                print("PDF solicitado (clic realizado)")
            return True
        except Exception as e:
            print("Error al intentar descargar/abrir PDF:", e)
            # intentar volver al primer handle si algo falló
            try:
                if self.driver.window_handles:
                    self.driver.switch_to.window(self.driver.window_handles[0])
            except Exception:
                pass
            return False
    
    
    def _norm(self, s: str) -> str:
        """Normaliza un título: quita acentos, puntuación, espacios extras y pasa a minúsculas."""
        if not s:
            return ""
        s = unicodedata.normalize("NFKD", s)
        s = "".join(ch for ch in s if not unicodedata.combining(ch))
        s = s.lower()
        s = re.sub(r"[^\w\s]", "", s)      # quitar puntuación
        s = re.sub(r"\s+", " ", s).strip() # colapsar espacios
        return s
    
    def should_skip(self, raw_title: str) -> bool:
        t = self._norm(raw_title)
        if t in self.SKIP_TITLES:
            return True
        return False

    def go_to_region_by_name(self, region_name: str, delay_between_actions: float = 1.0):
        """
        Localiza encabezado de región, recorre zonas, dentro de cada zona abre todas las rutas
        y descarga el PDF de cada ruta (si existe).
        """
        try:
            xpath_heading = (
                f"//a[contains(normalize-space(.), 'Rutas en {region_name}') "
                f"or normalize-space(.) = '{region_name}' "
                f"or contains(normalize-space(.), '{region_name}')]"
            )
            heading = self.wait.until(EC.presence_of_element_located((By.XPATH, xpath_heading)))
            self.driver.execute_script("arguments[0].scrollIntoView({block:'center'});", heading)
            time.sleep(0.25)

            zones_xpath = f"{xpath_heading}/following::a[contains(@class,'link_rojo_14_underline')]"
            zone_elements = self.driver.find_elements(By.XPATH, zones_xpath)
            
            print(f"Buscando zonas en la región '{region_name}'...")
            print(f"Se encontraron {len(zone_elements)} zonas:")

            for idx, zone in enumerate(zone_elements, start=1):
                zone_text = zone.text.strip()
                zone_href = zone.get_attribute("href")
                print(f" {zone_href}")
            
            if not zone_elements:
                print(f"No se encontraron zonas para la región '{region_name}'")
                return

            print(f"Se encontraron {len(zone_elements)} zonas en '{region_name}'")

            for zi in range(len(zone_elements)):
                heading = self.wait.until(EC.presence_of_element_located((By.XPATH, xpath_heading)))
                zone_elements = self.driver.find_elements(By.XPATH, zones_xpath)
                if zi >= len(zone_elements):
                    break
                zone_el = zone_elements[zi]
                zone_text = zone_el.text.strip()
                print(f"Procesando zona {zi+1}/{len(zone_elements)}: {zone_text}")

                self._click(zone_el)
                time.sleep(delay_between_actions)

                routes_selector = "a.content_ruta_zona"
                try:
                    self.wait.until(EC.presence_of_all_elements_located((By.CSS_SELECTOR, routes_selector)))
                    route_elements = self.driver.find_elements(By.CSS_SELECTOR, routes_selector)
                except Exception:
                    route_elements = []
                print(f"  Encontradas {len(route_elements)} rutas en zona '{zone_text}'")

                for ri in range(len(route_elements)):
                    try:
                        self.wait.until(EC.presence_of_all_elements_located((By.CSS_SELECTOR, routes_selector)))
                        route_elements = self.driver.find_elements(By.CSS_SELECTOR, routes_selector)
                    except Exception:
                        route_elements = []
                    if ri >= len(route_elements):
                        break
                    route_el = route_elements[ri]
                    route_title = route_el.text.strip()
                    norm_title = self._norm(route_title)
                    # Si el título normalizado está en la lista de skip, saltar sin abrir la ruta

                    if self.should_skip(route_title):
                        print(f"[SKIP] {route_title}")
                        ri += 1
                        continue

                    print(f"    Abriendo ruta {ri+1}/{len(route_elements)}: {route_title}")

                    self._click(route_el)
                    time.sleep(delay_between_actions)
                    # Extraer todos los detalles y guardar en CSV
                    route_data = self.get_route_details(route_title)
                    print(route_data)
                    if route_data:
                        self.save_route_to_csv(route_data,"rutas.csv")

                    # buscar y descargar PDF dentro del bloque #pdf_ruta
                    pdf_btn = self.wait_for_pdf_button(timeout=6)
                    if pdf_btn:
                        ok = self.download_pdf(pdf_btn)
                        if not ok:
                            print("Falló la descarga del PDF para:", route_title)
                    else:
                        print("Botón PDF no encontrado para la ruta:", route_title)

                    # volver atrás a la página de la zona para seguir con la siguiente ruta
                    try:
                        self.driver.back()
                        time.sleep(0.5)
                    except Exception as e:
                        print("Error al volver atrás después de la ruta:", e)

                # al terminar rutas de la zona, volver atrás a la lista de zonas
                try:
                    self.driver.back()
                    time.sleep(0.5)
                except Exception:
                    pass

            print("Procesado completo de la región:", region_name)
        except Exception as e:

            print("No se pudo localizar la región", region_name, e)


    def process_zone_by_link(self, zone_url: str, region_name: str = "", delay_between_actions: float = 1.2):
        """
        Procesa todas las rutas de una zona a partir de su URL directa.
        - zone_url: URL de la zona (ej: https://www.rutaspirineos.org/rutas/el-ripolles)
        """
        try:
            print(f"Abriendo zona desde URL: {zone_url}")
            self.driver.get(zone_url)
            time.sleep(2)

            # Extraer nombre de la zona de la URL si es necesario
            zone_name = zone_url.split('/')[-1].replace('-', ' ').title()
            print(f"Procesando zona: {zone_name}")

            # Esperar y obtener rutas de la zona
            routes_selector = "a.content_ruta_zona"
            try:
                self.wait.until(EC.presence_of_all_elements_located((By.CSS_SELECTOR, routes_selector)))
                route_elements = self.driver.find_elements(By.CSS_SELECTOR, routes_selector)
            except Exception:
                route_elements = []

            print(f"Encontradas {len(route_elements)} rutas en zona '{zone_name}'")

            # Recorrer rutas
            for ri in range(len(route_elements)):
                try:
                    self.wait.until(EC.presence_of_all_elements_located((By.CSS_SELECTOR, routes_selector)))
                    route_elements = self.driver.find_elements(By.CSS_SELECTOR, routes_selector)
                except Exception:
                    route_elements = []

                if ri >= len(route_elements):
                    break

                route_el = route_elements[ri]
                route_title = route_el.text.strip()
                norm_title = self._norm(route_title)
                route_href = route_el.get_attribute("href") or ""
                if "rutas-guiadas" in route_href.lower():
                    print(f"  [SKIP GUIADA] {route_title} -> {route_href}")
                    continue

                print(f"  Abriendo ruta {ri+1}/{len(route_elements)}: {route_title}")

                self._click(route_el)
                time.sleep(delay_between_actions)

                # Extraer detalles de la ruta
                route_data = self.get_route_details(route_title)
                if route_data:
                    # Buscar y descargar PDF
                    pdf_btn = self.wait_for_pdf_button(timeout=6)
                    pdf_path = ""
                    if pdf_btn:
                        pdf_path = self.download_pdf(pdf_button=pdf_btn, wait_after_click=0.6)
                        if pdf_path:
                            route_data["Pdf_path"] = pdf_path
                        else:
                            print(f"Falló la descarga del PDF para: {route_title}")
                    else:
                        print(f"Botón PDF no encontrado para: {route_title}")

                    # Guardar en CSV
                    self.save_route_to_csv(route_data, "rutas_pirineos.csv")

                # Volver atrás a la página de la zona
                try:
                    self.driver.back()
                    time.sleep(0.5)
                except Exception as e:
                    print(f"Error al volver atrás: {e}")

            print(f"Procesado completo de la zona: {zone_name}")

        except Exception as e:
            print(f"Error procesando zona desde URL: {e}")


if __name__ == "__main__":
    scraper = Scraper(headless=False)
    try:
        scraper.start("https://www.rutaspirineos.org/")
        time.sleep(1)
        # Gestionar subscripción
        scraper.handle_subscription(subscribe=True, name="Jessica", email="jessicaromero8100@uoc.edu")
        scraper.click_guides_menu()
        time.sleep(3)
        # ejemplo: procesar la región "Aragón" (usa el texto tal cual aparece)
        # scraper.go_to_region_by_name("Cataluña", delay_between_actions=1.2)
        # time.sleep(5)
        # Procesar zona específica desde URL
        
        #https://www.rutaspirineos.org/rutas/maresme
        zone_urls = ['https://www.rutaspirineos.org/rutas/parque-natural-cabeceras-ter-freser',
                    'https://www.rutaspirineos.org/rutas/parque-natural-de-la-zona-volcanica-de-la-garrotxa',
                    'https://www.rutaspirineos.org/rutas/parque-natural-de-las-marismas-del-ampurdan',
                    'https://www.rutaspirineos.org/rutas/parque-natural-de-los-valles-occidentales',
                    'https://www.rutaspirineos.org/rutas/parque-natural-de-posets-maladeta',
                    'https://www.rutaspirineos.org/rutas/parc-natural-alt-pirineu',
                    'https://www.rutaspirineos.org/rutas/parque-natural-del-cabo-de-creus',
                    'https://www.rutaspirineos.org/rutas/parque-natural-cadi-moixero',
                    'https://www.rutaspirineos.org/rutas/parque-natural-del-valle-de-sorteny',
                    'https://www.rutaspirineos.org/rutas/parc-naturel-regional-des-pyrenees-catalanes',
                    'https://www.rutaspirineos.org/rutas/parc-naturel-regional-des-pyrenees-ariegeoises',
                    'https://www.rutaspirineos.org/rutas/parque-natural-valle-de-madriu-perafita-claror',
                    'https://www.rutaspirineos.org/rutas/parque-natural-valles-del-comapedrosa']
        

        for idx, zone_url in enumerate(zone_urls, start=1):
            print(f"\n===== [{idx}/{len(zone_urls)}] Procesando zona: {zone_url} =====")
            try:
                scraper.process_zone_by_link(
                    zone_url=zone_url,
                    region_name="Cataluña",
                    delay_between_actions=1.0
                )
            except Exception as e:
                print(f"Error procesando {zone_url}: {e}")
                # continuar con la siguiente sin interrumpir el resto
                continue

        print("\n Procesamiento de todas las zonas completado.")
        #time.sleep(5)
    finally:
        scraper.quit()
