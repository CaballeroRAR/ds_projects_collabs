# src/data/extract.py
"""
Módulo para extraer datos del dataset de retail.
Convierte archivos Excel a formato pickle para procesamiento posterior.
"""

import pandas as pd
from pathlib import Path
from typing import List, Dict, Union, Optional
import warnings
warnings.filterwarnings('ignore')

class DataExtractor:
    """Clase para extraer y convertir datos de Excel a formatos procesables (Pickle/Parquet).

    Esta clase facilita la carga de archivos Excel de gran tamaño, permitiendo 
    seleccionar hojas específicas y guardarlas en formatos binarios rápidos 
    para su posterior análisis.

    Attributes:
        project_root (Path): Ruta raíz del proyecto.
        data_raw (Path): Ruta al archivo Excel original.
        data_processed (Path): Directorio de salida para archivos procesados.
    """
    
    def __init__(self, project_root: Optional[Path] = None, raw_filename: str = 'online_retail_II.xlsx'):
        """
        Inicializa el extractor de datos
        
        Args:
            project_root: Ruta raíz del proyecto. Si es None, se infiere.
            raw_filename: Nombre del archivo Excel original.
        """
        if project_root is None:
            # Inferencia robusta basada en la ubicación del archivo (src/data/extract.py)
            self.project_root = Path(__file__).resolve().parents[2]
        else:
            self.project_root = Path(project_root)
        
        self.raw_filename = raw_filename
        
        # Configurar rutas
        self.data_raw = self.project_root / 'data' / 'raw' / self.raw_filename
        self.data_processed = self.project_root / 'data' / 'processed'
        
        # Crear directorios si no existen
        self.data_processed.mkdir(parents=True, exist_ok=True)
        
        print(f"✅ DataExtractor inicializado:")
        print(f"   Proyecto: {self.project_root}")
        print(f"   Excel: {self.data_raw}")
        print(f"   Output: {self.data_processed}")
    
    def get_available_sheets(self) -> List[str]:
        """Obtiene la lista de nombres de todas las hojas en el archivo Excel.

        Returns:
            List[str]: Lista de nombres de las hojas encontradas.
            
        Example:
            >>> extractor = DataExtractor()
            >>> sheets = extractor.get_available_sheets()
            >>> print(sheets)
            ['Year 2009-2010', 'Year 2010-2011']
        """
        if not self.data_raw.exists():
            print(f"❌ Archivo no encontrado: {self.data_raw}")
            return []
        
        try:
            xls = pd.ExcelFile(self.data_raw)
            sheets = xls.sheet_names
            
            print(f"📋 Hojas disponibles ({len(sheets)}):")
            for i, sheet in enumerate(sheets, 1):
                print(f"   {i}. {sheet}")
            
            return sheets
            
        except Exception as e:
            print(f"❌ Error al leer Excel: {e}")
            return []
    
    def load_sheet(self, sheet_name: str) -> pd.DataFrame:
        """Carga una hoja específica del Excel en un DataFrame.

        Args:
            sheet_name (str): Nombre exacto de la hoja a cargar.
        
        Returns:
            pd.DataFrame: DataFrame con los datos de la hoja. Si hay error, devuelve un DF vacío.
        """
        print(f"📥 Cargando hoja: '{sheet_name}'")
        
        try:
            df = pd.read_excel(
                self.data_raw,
                sheet_name=sheet_name,
                dtype={'Invoice': str, 'StockCode': str, 'Customer ID': str}
            )
            
            print(f"   ✅ Cargado: {df.shape[0]:,} filas, {df.shape[1]} columnas")
            return df
            
        except Exception as e:
            print(f"❌ Error al cargar '{sheet_name}': {e}")
            return pd.DataFrame()
    
    def load_all_sheets(self) -> Dict[str, pd.DataFrame]:
        """
        Carga TODAS las hojas del Excel
        
        Returns:
            Diccionario con DataFrames por nombre de hoja
        """
        print("📥 Cargando TODAS las hojas...")
        
        if not self.data_raw.exists():
            print(f"❌ Archivo no encontrado: {self.data_raw}")
            return {}
        
        try:
            # sheet_name=None es clave para cargar todas las hojas
            dfs = pd.read_excel(
                self.data_raw,
                sheet_name=None,  # ¡IMPORTANTE! None = todas las hojas
                dtype={'Invoice': str, 'StockCode': str, 'Customer ID': str}
            )
            
            print(f"✅ Cargadas {len(dfs)} hojas:")
            for sheet_name, df in dfs.items():
                print(f"   • {sheet_name}: {df.shape[0]:,} filas")
            
            return dfs
            
        except Exception as e:
            print(f"❌ Error al cargar Excel: {e}")
            return {}
    
    def save_to_pkl(self, 
                    data: Union[pd.DataFrame, Dict[str, pd.DataFrame]], 
                    base_name: str,
                    save_individual: bool = True) -> List[Path]:
        """
        Guarda datos en formato pickle
        
        Args:
            data: DataFrame o dict de DataFrames
            base_name: Nombre base para los archivos
            save_individual: Si es True, guarda cada hoja individualmente
        
        Returns:
            Lista de rutas de archivos guardados
        """
        saved_paths = []
        
        if isinstance(data, dict):
            # Diccionario de DataFrames (múltiples hojas)
            print(f"💾 Guardando {len(data)} archivo(s) PKL...")
            
            if save_individual:
                for sheet_name, df in data.items():
                    safe_name = sheet_name.lower().replace(' ', '_').replace('-', '_')
                    filename = f"{base_name}_{safe_name}.pkl"
                    filepath = self.data_processed / filename
                    
                    df.to_pickle(filepath)
                    saved_paths.append(filepath)
                    print(f"   ✅ {filename} ({df.shape[0]:,} filas)")
            
            # Guardar combinado si hay más de una hoja
            if len(data) > 1:
                combined_df = pd.concat(data.values(), ignore_index=True)
                combined_filename = f"{base_name}_combined.pkl"
                combined_path = self.data_processed / combined_filename
                
                combined_df.to_pickle(combined_path)
                saved_paths.append(combined_path)
                print(f"   ✅ {combined_filename} ({combined_df.shape[0]:,} filas)")
        
        elif isinstance(data, pd.DataFrame):
            # DataFrame único
            filename = f"{base_name}.pkl"
            filepath = self.data_processed / filename
            
            data.to_pickle(filepath)
            saved_paths.append(filepath)
            print(f"💾 Guardado: {filename} ({data.shape[0]:,} filas)")
        
        else:
            print(f"❌ Tipo de datos no soportado: {type(data)}")
        
        print(f"\n📁 Archivos guardados en: {self.data_processed}")
        return saved_paths
    
    def extract_specific_sheets(self, 
                               sheet_names: List[str], 
                               output_name: Optional[str] = None) -> Dict[str, pd.DataFrame]:
        """
        Extrae hojas específicas y las guarda como PKL
        
        Args:
            sheet_names: Lista de nombres de hojas
            output_name: Nombre base para los archivos (si es None, se genera automáticamente)
        
        Returns:
            Diccionario con DataFrames cargados
        """
        print(f"🎯 Extrayendo {len(sheet_names)} hoja(s) específica(s)...")
        
        if output_name is None:
            # Generar nombre automático
            safe_names = [s.replace(" ", "_").replace("-", "_").lower() for s in sheet_names]
            output_name = "_".join(safe_names[:3])  # Máximo 3 nombres
        
        dfs = {}
        
        for sheet in sheet_names:
            df = self.load_sheet(sheet)
            if not df.empty:
                dfs[sheet] = df
        
        if dfs:
            self.save_to_pkl(dfs, output_name, save_individual=True)
        else:
            print("❌ No se pudo cargar ninguna hoja")
        
        return dfs
    
    def extract_all_sheets(self, output_name: str = "retail_all") -> Dict[str, pd.DataFrame]:
        """
        Extrae TODAS las hojas y las guarda como PKL
        
        Args:
            output_name: Nombre base para los archivos
        
        Returns:
            Diccionario con todas las hojas
        """
        print("🎯 Extrayendo TODAS las hojas...")
        
        dfs = self.load_all_sheets()
        
        if dfs:
            self.save_to_pkl(dfs, output_name, save_individual=True)
        else:
            print("❌ No se pudo cargar ninguna hoja")
        
        return dfs
    
    def list_pkl_files(self) -> List[Path]:
        """
        Lista todos los archivos PKL en el directorio processed
        
        Returns:
            Lista de rutas a archivos PKL
        """
        pkl_files = sorted(self.data_processed.glob("*.pkl"))
        
        if pkl_files:
            print(f"📁 Archivos PKL disponibles ({len(pkl_files)}):")
            for i, pkl_file in enumerate(pkl_files, 1):
                size_mb = pkl_file.stat().st_size / (1024 * 1024)
                print(f"   {i}. {pkl_file.name} ({size_mb:.2f} MB)")
        else:
            print("📭 No hay archivos PKL en el directorio")
        
        return pkl_files
    
    def load_pkl(self, pattern: str = "*.pkl") -> Dict[str, pd.DataFrame]:
        """
        Carga archivos PKL existentes
        
        Args:
            pattern: Patrón para buscar archivos (ej: "*2010*.pkl")
        
        Returns:
            Diccionario con DataFrames cargados
        """
        pkl_files = sorted(self.data_processed.glob(pattern))
        
        if not pkl_files:
            print(f"📭 No se encontraron archivos con patrón '{pattern}'")
            return {}
        
        print(f"📂 Cargando {len(pkl_files)} archivo(s) PKL...")
        
        dfs = {}
        for pkl_file in pkl_files:
            try:
                df = pd.read_pickle(pkl_file)
                dfs[pkl_file.name] = df
                print(f"   ✅ {pkl_file.name}: {df.shape[0]:,} filas")
            except Exception as e:
                print(f"   ❌ {pkl_file.name}: Error - {e}")
        
        return dfs


# Función de conveniencia para uso rápido
def extract_data(sheets="all", output_name="retail_data", project_root=None):
    """
    Función de conveniencia para extraer datos rápidamente
    
    Args:
        sheets: "all" o lista de nombres de hojas
        output_name: Nombre base para los archivos PKL
        project_root: Ruta raíz del proyecto
    
    Returns:
        Diccionario con DataFrames
    """
    extractor = DataExtractor(project_root)
    
    if sheets == "all":
        return extractor.extract_all_sheets(output_name)
    else:
        return extractor.extract_specific_sheets(sheets, output_name)


def load_raw_dataset(project_root=None) -> pd.DataFrame:
    """Convenience function to load all sheets from raw Excel as a single DataFrame."""
    extractor = DataExtractor(project_root)
    dfs = extractor.load_all_sheets()
    if not dfs:
        return pd.DataFrame()
    return pd.concat(dfs.values(), ignore_index=True)


# Uso principal del script
if __name__ == "__main__":
    print("\n" + "🚀" + " " + "Iniciando proceso de extracción".center(48, "=") + " 🚀")
    
    # 1. Inicializar extractor
    extractor = DataExtractor()
    
    # 2. Listar hojas disponibles
    sheets = extractor.get_available_sheets()
    
    if sheets:
        # 3. Ejemplo: Extraer solo la primera hoja
        print(f"\n📝 Ejemplo 1: Extrayendo solo '{sheets[0]}'")
        extractor.extract_specific_sheets([sheets[0]], output_name="retail_single_sheet")
        
        # 4. Ejemplo: Extraer y combinar todas las hojas
        print("\n📝 Ejemplo 2: Extrayendo y combinando todas las hojas")
        data = extractor.extract_all_sheets("retail_full_dataset")
        
        # 5. Listar resultados
        print("\n" + "="*50)
        extractor.list_pkl_files()
    
    print("\n" + "✅ Procesamiento finalizado con éxito ".center(50, "="))