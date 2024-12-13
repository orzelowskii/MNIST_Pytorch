# Klasyfikacja cyfr z plików CSV przy użyciu PyTorch
## Opis projektu
Ten projekt dotyczy budowy modelu klasyfikacji cyfr (0-9) na podstawie danych zapisanych w plikach CSV, reprezentujących obrazy cyfr. Projekt został zrealizowany przy użyciu frameworka PyTorch, a główne cele to:

1. Wczytanie i przetworzenie danych z plików CSV, które reprezentują obrazy.
2. Stworzenie modelu sieci neuronowej z wykorzystaniem PyTorch.
3. Trening i ocena modelu na podzielonych zbiorach danych (treningowym i testowym).
4. Wizualizacja rozkładu klas oraz analiza wyników.

Projekt pokazuje pełny przepływ pracy, od przygotowania danych, przez budowę i trening modelu, po analizę wyników.

## Funkcjonalności
- Przetwarzanie danych: Dane wejściowe to obrazy cyfr zapisane w formacie CSV. Każdy wiersz reprezentuje jeden obraz w formie wektorów pikseli oraz przypisaną klasę.
- Budowa modelu: Model to prosta, ale skuteczna sieć neuronowa oparta na dwóch warstwach w pełni połączonych (Linear) z funkcją aktywacji ReLU.
- Trening i testowanie: Model jest trenowany na zbiorze treningowym i oceniany na zbiorze testowym z wykorzystaniem funkcji kosztu CrossEntropyLoss i optymalizatora Adam.
- Wizualizacja: Stworzono wykresy prezentujące procentowy rozkład klas w zbiorze danych oraz analizę wyników modelu.
- Raport wyników: Wyliczono wskaźniki, takie jak dokładność klasyfikacji.
