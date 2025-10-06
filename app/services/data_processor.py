import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import os
import glob

class ExoplanetDataProcessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_columns = []
        # Mapeo de columnas KOI a formato estándar
        self.column_mapping = {
            'koi_period': 'pl_orbper',
            'koi_duration': 'pl_trandurh',
            'koi_depth': 'pl_trandep',
            'koi_prad': 'pl_rade',
            'koi_insol': 'pl_insol',
            'koi_teq': 'pl_eqt',
            'koi_kepmag': 'st_tmag',
            'koi_steff': 'st_teff',
            'koi_slogg': 'st_logg',
            'koi_srad': 'st_rad'
        }

    def load_data(self, data_path="data/"):
        """Load and preprocess the exoplanet data from all CSV files in the data folder"""
        try:
            # Find all CSV files in the data directory
            if os.path.isdir(data_path):
                csv_files = glob.glob(os.path.join(data_path, "*.csv"))
                if not csv_files:
                    raise Exception(f"No CSV files found in {data_path}")

                print(f"Found {len(csv_files)} CSV files: {[os.path.basename(f) for f in csv_files]}")

                # Load and combine all CSV files
                dataframes = []
                for csv_file in csv_files:
                    try:
                        df = pd.read_csv(csv_file, comment='#')
                        # Renombrar columnas KOI al formato estándar si existen
                        df = df.rename(columns=self.column_mapping)
                        
                        # Manejar columna de disposición para archivos KOI
                        if 'koi_disposition' in df.columns and 'tfopwg_disp' not in df.columns:
                            # Mapear valores de koi_disposition a formato tfopwg_disp
                            df['tfopwg_disp'] = df['koi_disposition'].map({
                                'CONFIRMED': 'CP',
                                'CANDIDATE': 'PC',
                                'FALSE POSITIVE': 'FP'
                            })
                            print(f"  Mapped koi_disposition to tfopwg_disp")
                        
                        dataframes.append(df)
                        print(f"Loaded {len(df)} rows from {os.path.basename(csv_file)}")
                    except Exception as file_error:
                        print(f"Warning: Could not load {csv_file}: {str(file_error)}")
                        continue

                if not dataframes:
                    raise Exception("No valid CSV files could be loaded")

                # Combine all dataframes
                df = pd.concat(dataframes, ignore_index=True)
                print(f"Combined dataset: {len(df)} total rows")

            else:
                # Single file path provided
                df = pd.read_csv(data_path, comment='#')
                # Renombrar columnas KOI al formato estándar si existen
                df = df.rename(columns=self.column_mapping)
                
                # Manejar columna de disposición para archivos KOI
                if 'koi_disposition' in df.columns and 'tfopwg_disp' not in df.columns:
                    # Mapear valores de koi_disposition a formato tfopwg_disp
                    df['tfopwg_disp'] = df['koi_disposition'].map({
                        'CONFIRMED': 'CP',
                        'CANDIDATE': 'PC',
                        'FALSE POSITIVE': 'FP'
                    })
                    print(f"  Mapped koi_disposition to tfopwg_disp")
                
                print(f"Loaded single file: {len(df)} rows")

            # Select relevant numerical features for classification
            feature_cols = [
                'pl_orbper', 'pl_trandurh',
                'pl_trandep', 'pl_rade', 'pl_insol', 'pl_eqt', 'st_tmag',
                'st_teff', 'st_logg', 'st_rad'
            ]

            # Filter columns that exist in the dataset
            available_features = [col for col in feature_cols if col in df.columns]
            self.feature_columns = available_features
            print(f"Using {len(available_features)} features: {available_features}")

            # Extract features
            X = df[available_features].copy()

            # Create target variable based on TFOPWG disposition
            # CP (Confirmed Planet) = 1 (Exoplanet)
            # FP (False Positive), KP (Known Planet), PC (Planet Candidate) = 0 (Not confirmed exoplanet)
            if 'tfopwg_disp' not in df.columns:
                raise Exception("Column 'tfopwg_disp' not found in dataset. This column is required for classification.")

            y = (df['tfopwg_disp'] == 'CP').astype(int)

            # Handle missing values
            X = X.fillna(X.median())

            # Remove rows with all NaN values in target
            valid_mask = ~pd.isna(df['tfopwg_disp'])
            X = X[valid_mask]
            y = y[valid_mask]

            # Remove duplicates based on all features
            combined_data = pd.concat([X, y], axis=1)
            combined_data = combined_data.drop_duplicates()
            X = combined_data[available_features]
            y = combined_data['tfopwg_disp']

            print(f"Final dataset after preprocessing: {len(X)} rows")
            print(f"Confirmed exoplanets: {sum(y)} ({sum(y)/len(y)*100:.2f}%)")
            print(f"Non-exoplanets: {len(y) - sum(y)} ({(len(y) - sum(y))/len(y)*100:.2f}%)")

            return X, y

        except Exception as e:
            raise Exception(f"Error loading data: {str(e)}")

    def preprocess_data(self, X, y, test_size=0.2, random_state=42):
        """Preprocess the data for training"""
        try:
            # Split the data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state, stratify=y
            )

            # Scale the features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)

            return X_train_scaled, X_test_scaled, y_train, y_test

        except Exception as e:
            raise Exception(f"Error preprocessing data: {str(e)}")

    def get_data_info(self):
        """Get information about the loaded data"""
        try:
            # Get list of CSV files
            data_path = "data/"
            csv_files = []
            if os.path.isdir(data_path):
                csv_files = glob.glob(os.path.join(data_path, "*.csv"))
                csv_files = [os.path.basename(f) for f in csv_files]

            X, y = self.load_data()

            total_samples = len(X)
            exoplanets = sum(y)
            non_exoplanets = total_samples - exoplanets

            return {
                "data_source": {
                    "csv_files_found": csv_files,
                    "total_files": len(csv_files),
                    "data_directory": data_path
                },
                "dataset_statistics": {
                    "total_samples": total_samples,
                    "confirmed_exoplanets": exoplanets,
                    "non_exoplanets": non_exoplanets,
                    "exoplanet_percentage": round((exoplanets / total_samples) * 100, 2)
                },
                "features": {
                    "features_used": self.feature_columns,
                    "feature_count": len(self.feature_columns)
                }
            }

        except Exception as e:
            return {"error": f"Error getting data info: {str(e)}"}

    def prepare_single_prediction(self, data_dict):
        """Prepare a single sample for prediction"""
        try:
            # Ensure feature columns are loaded
            if not self.feature_columns:
                # Load data to initialize feature columns
                self.load_data()

            # Create a DataFrame with the same structure as training data
            sample_df = pd.DataFrame([data_dict])

            # Select only the features used in training, fill missing with 0
            available_features = [col for col in self.feature_columns if col in sample_df.columns]
            sample_features = sample_df[available_features].fillna(0)

            # Add missing features as zeros
            for feature in self.feature_columns:
                if feature not in sample_features.columns:
                    sample_features[feature] = 0.0

            # Reorder columns to match training order
            sample_features = sample_features[self.feature_columns]

            # Scale the features using the fitted scaler
            sample_scaled = self.scaler.transform(sample_features)

            return sample_scaled

        except Exception as e:
            raise Exception(f"Error preparing prediction data: {str(e)}")