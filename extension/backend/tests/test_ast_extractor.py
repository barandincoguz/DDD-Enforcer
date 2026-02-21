import unittest
import tempfile
import os
import shutil
from pathlib import Path

# Sınıfı import ediyoruz (Dosya yoluna göre import'u düzenlemen gerekebilir)
# Eğer bu dosyayı backend/tests/ içine koyduysan ve ana klasör backend ise:
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.ast_model_signals import ASTModelSignalExtractor
from core.schemas import DomainModel

class TestASTSignalExtractor(unittest.TestCase):

    def setUp(self):
        # Her testten önce geçici bir klasör oluştur
        self.test_dir = tempfile.mkdtemp()
        self.extractor = ASTModelSignalExtractor()

    def tearDown(self):
        # Her testten sonra temizlik yap
        shutil.rmtree(self.test_dir)

    def create_file(self, filename, content):
        path = os.path.join(self.test_dir, filename)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        return path

    def test_value_object_detection_no_hardcoding(self):
        """
        TEST 1: İsimde 'Money' veya 'Address' geçmese bile yapısal olarak
        Value Object (Değer Nesnesi) olduğunu anlamalı.
        """
        code = """
from dataclasses import dataclass

@dataclass(frozen=True)
class GpsCoordinate:
    latitude: float
    longitude: float
    
    def distance_to(self, other):
        pass
"""
        self.create_file("vo.py", code)
        results = self.extractor.extract_candidates([os.path.join(self.test_dir, "vo.py")])
        
        # Kontrol: GpsCoordinate bir VO olarak algılandı mı?
        vo_names = [item["name"] for item in results["value_objects"]]
        self.assertIn("GpsCoordinate", vo_names, "Frozen dataclass Value Object olarak algılanmalıydı.")
        print("✅ Value Object Testi (İsimden bağımsız) Başarılı")

    def test_service_detection_via_dependency_injection(self):
        """
        TEST 2: Suffix 'Service' olmasa bile, Constructor'da dependency 
        almasından dolayı bunun bir servis olduğunu anlamalı.
        """
        code = """
class OrderProcessor:
    def __init__(self, order_repository, email_sender):
        self.repo = order_repository
        self.mailer = email_sender
        
    def process_payment(self, order_id):
        # İş mantığı
        pass
"""
        self.create_file("service.py", code)
        results = self.extractor.extract_candidates([os.path.join(self.test_dir, "service.py")])
        
        service_names = [item["name"] for item in results["services"]]
        self.assertIn("OrderProcessor", service_names, "Dependency Injection kullanan sınıf Service olarak algılanmalıydı.")
        print("✅ Service Testi (DI Analizi) Başarılı")

    def test_entity_detection_with_identity(self):
        """
        TEST 3: Bir sınıfın Entity olduğunu 'id' alanı ve metotlarından anlamalı.
        """
        code = """
class Customer:
    def __init__(self, customer_id, name):
        self.customer_id = customer_id
        self.name = name
        self.is_active = True
        
    def activate(self):
        self.is_active = True
        
    def deactivate(self):
        self.is_active = False
"""
        self.create_file("entity.py", code)
        results = self.extractor.extract_candidates([os.path.join(self.test_dir, "entity.py")])
        
        entity_names = [item["name"] for item in results["entities"]]
        self.assertIn("Customer", entity_names, "ID ve metotları olan sınıf Entity olarak algılanmalıydı.")
        print("✅ Entity Testi Başarılı")

    def test_aggregate_detection(self):
        """
        TEST 4: Koleksiyon yöneten sınıfı Aggregate Root olarak algılamalı.
        """
        code = """
class ShoppingCart:
    def __init__(self):
        self.cart_id = "123"
        self.items = [] # Koleksiyon
        
    def add_item(self, item):
        self.items.append(item)
        
    def remove_item(self, item_id):
        pass
"""
        self.create_file("aggregate.py", code)
        results = self.extractor.extract_candidates([os.path.join(self.test_dir, "aggregate.py")])
        
        agg_names = [item["name"] for item in results["aggregates"]]
        self.assertIn("ShoppingCart", agg_names, "Koleksiyon yöneten sınıf Aggregate olarak algılanmalıydı.")
        print("✅ Aggregate Testi Başarılı")

    def test_false_positive_elimination(self):
        """
        TEST 5: DTO'lar veya Helper sınıfları Entity sanılmamalı.
        """
        code = """
@dataclass
class UserDTO:
    username: str
    password: str

class StringHelper:
    @staticmethod
    def to_upper(s):
        return s.upper()
"""
        self.create_file("misc.py", code)
        results = self.extractor.extract_candidates([os.path.join(self.test_dir, "misc.py")])
        
        entity_names = [item["name"] for item in results["entities"]]
        self.assertNotIn("UserDTO", entity_names, "DTO entity olmamalı.")
        self.assertNotIn("StringHelper", entity_names, "Helper entity olmamalı.")
        print("✅ False Positive (Yanlış Tespit) Testi Başarılı")

if __name__ == "__main__":
    unittest.main()