## ----------------------------------------------------------
## ŁADOWANIE BIBLIOTEK
## ----------------------------------------------------------
library(forecast)
library(zoo)         
library(dplyr)       
library(lubridate)   
library(ggplot2)
library(tseries)
library(reshape2)
library(kableExtra)
## ----------------------------------------------------------
## WPROWADZENIE DANYCH
## ----------------------------------------------------------

# Wczytaj dane z GitHuba (własna ścieżka projektu)
project_df <- read.csv("https://raw.githubusercontent.com/Julia6400/Projekt_Rstudio_szeregiczasowe/main/dane/Walmart%20Qtr.csv", header = TRUE, stringsAsFactors = FALSE)

# Konwersja kolumny daty
project_df$Date <- as.Date(project_df$Date, format = "%m/%d/%y")
project_df$Quarter <- paste0(year(project_df$Date), " Q", quarter(project_df$Date))


# Zmiana nazw kolumn
project_df <- project_df %>% rename(
  Przychod = `Revenue.Millions.of.US...`
)

# Usunięcie przecinków i konwersja przychodu na liczbę
project_df$Przychod <- as.numeric(gsub(",", "", project_df$Przychod))

# Sortowanie i wybór kolumn
project_df <- project_df %>% arrange(Date) %>% select(Quarter, Przychod)

# Przekształcenie kwartałów w faktory
project_df$Quarter <- factor(project_df$Quarter, levels = unique(project_df$Quarter))

# Podstawowe podsumowanie danych
summary(project_df)

# Wizualizacja przychodu kwartalnego
ggplot(project_df, aes(x = Quarter, y = Przychod, group = 1)) +
  geom_line(color = "darkgreen", linewidth = 1) +  
  geom_point(color = "black") +
  labs(title = "Kwartalny przychód Walmart", x = "Kwartał", y = "Przychód (mln USD)") +
  scale_x_discrete(breaks = project_df$Quarter[seq(1, length(project_df$Quarter), by = 4)]) +
  theme_minimal() +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1),
    panel.grid.major = element_line(color = "gray", size = 0.3),
    panel.grid.minor = element_line(color = "gray", size = 0.2)
  )

## ----------------------------------------------------------
## TWORZENIE SZEREGU CZASOWEGO
## ----------------------------------------------------------
walmart.ts <- ts(project_df$Przychod, start = c(2009, 1), end = c(2024, 4), freq = 4)

## ----------------------------------------------------------
## PODZIAŁ NA ZBIÓR TRENINGOWY I WALIDACYJNY
## ----------------------------------------------------------
nValid <- 16
nTrain <- length(walmart.ts) - nValid
train.ts <- window(walmart.ts, start = c(2009, 1), end = c(2009, nTrain))
valid.ts <- window(walmart.ts, start = c(2009, nTrain + 1), end = c(2009, nTrain + nValid))

# Wykres ACF
Acf(walmart.ts, lag.max = 12, main = "Autokorelacja kwartalnych przychodów Walmart")

# Wykres z podziałem na zbiór treningowy i walidacyjny
plot(walmart.ts, xlab = "Rok", ylab = "Przychód (mln USD)",
     main = "Przychód kwartalny Walmart – dane treningowe i walidacyjne", lwd = 2, xaxt = "n")
axis(1, at = seq(2009, 2024, by = 1), labels = seq(2009, 2024, by = 1))
lines(train.ts, col = "blue", lwd = 2)
abline(v = time(train.ts)[length(train.ts)], col = "red", lwd = 2, lty = 2)
legend("topleft", legend = c("Trening", "Walidacja"), col = c("blue", "black"), lty = 1, lwd = 2, bty = "n")

# Dekompzycja STL
walmart.stl <- stl(walmart.ts, s.window = "periodic")
autoplot(walmart.stl, main = "STL – dekompozycja przychodów Walmart")

## ----------------------------------------------------------
## TEST ADF I RÓŻNICOWANIE
## ----------------------------------------------------------
adf.test(walmart.ts)

diff_walmart <- diff(walmart.ts, differences = 1)
adf.test(diff_walmart)

diff_seasonal_walmart <- diff(diff_walmart, lag = 4)
adf.test(diff_seasonal_walmart)

Acf(diff_seasonal_walmart, lag.max = 12, main = "Autokorelacja po różnicowaniu sezonowym")

## ----------------------------------------------------------
## MODELE PROGNOZUJĄCE: ARIMA I HOLT-WINTERS
## ----------------------------------------------------------

# Model ARIMA
model_arima <- auto.arima(train.ts)
summary(model_arima)

# Sprawdzenie reszt modelu
checkresiduals(model_arima)

# Prognoza ARIMA na zbiorze walidacyjnym
forecast_arima <- forecast(model_arima, h = nValid)

# Wykres prognozy ARIMA
plot(forecast_arima, xlab = "Rok", ylab = "Przychód (mln USD)",
     main = "Prognoza przychodu Walmart – model ARIMA", ylim = c(95000, 180000),
     xlim = c(2009, 2025), lwd = 2)
lines(valid.ts, col = "black", lwd = 2)
abline(v = time(train.ts)[length(train.ts)], col = "red", lty = 2, lwd = 2)
legend("topleft", legend = c("Dane rzeczywiste", "Prognoza ARIMA"),
       col = c("black", "blue"), lty = 1, lwd = 2, bty = "n")

# Model Holt-Winters
model_hw <- ets(train.ts, model = "ZZZ")
summary(model_hw)

# Prognoza Holt-Winters na zbiorze walidacyjnym
forecast_hw <- forecast(model_hw, h = nValid)

# Wykres prognozy Holt-Winters
plot(forecast_hw, xlab = "Rok", ylab = "Przychód (mln USD)",
     main = "Prognoza przychodu Walmart – model Holt-Winters", ylim = c(95000, 180000),
     xlim = c(2009, 2026), lwd = 2)
lines(valid.ts, col = "black", lwd = 2)
abline(v = time(train.ts)[length(train.ts)], col = "red", lty = 2, lwd = 2)
legend("topleft", legend = c("Dane rzeczywiste", "Prognoza Holt-Winters"),
       col = c("black", "blue"), lty = 1, lwd = 2, bty = "n")

## ----------------------------------------------------------
## EWALUACJA MODELI NA ZBIORZE WALIDACYJNYM
## ----------------------------------------------------------
accuracy_arima <- round(accuracy(forecast_arima$mean, valid.ts), 2)
accuracy_hw <- round(accuracy(forecast_hw$mean, valid.ts), 2)

print("Dokładność prognozy – model ARIMA:")
print(accuracy_arima)

print("Dokładność prognozy – model Holt-Winters:")
print(accuracy_hw)


## ----------------------------------------------------------
## KOŃCOWA PROGNOZA NA PRZYSZŁOŚĆ
## ----------------------------------------------------------
final_model <- if (accuracy_arima["Test set", "RMSE"] < accuracy_hw["Test set", "RMSE"]) model_arima else model_hw

final_forecast <- forecast(final_model, h = 16)

plot(final_forecast, xlab = "Rok", ylab = "Przychód (mln USD)",
     main = "Prognoza przychodu Walmart na 16 kwartałów wprzód",
     xlim = c(2009, 2030), ylim = c(80000, 220000), lwd = 2)
legend("topleft", legend = c("Prognoza na przyszłość"),
       col = c("blue"), lty = 1, lwd = 2, bty = "n")


## ----------------------------------------------------------
## PORÓWNANIE MODELI – TABELA PODSUMOWUJĄCA
## ----------------------------------------------------------
# Tworzenie tabeli z metrykami RMSE i MAPE
porownanie <- data.frame(
  Model = c("ARIMA", "Holt-Winters"),
  RMSE = c(accuracy_arima["Test set", "RMSE"], accuracy_hw["Test set", "RMSE"]),
  MAPE = c(accuracy_arima["Test set", "MAPE"], accuracy_hw["Test set", "MAPE"])
)

# Podświetlenie najlepszych wyników
porownanie_wiz <- porownanie %>%
  mutate(
    RMSE = cell_spec(RMSE, format = "html",
                     color = ifelse(RMSE == min(RMSE), "green", "black"),
                     bold = RMSE == min(RMSE)),
    MAPE = cell_spec(MAPE, format = "html",
                     color = ifelse(MAPE == min(MAPE), "green", "black"),
                     bold = MAPE == min(MAPE))
  )

# Tabela HTML do raportu
kable(porownanie_wiz, escape = FALSE, format = "html",
      caption = "Porównanie dokładności modeli prognozujących (najlepsze wartości zaznaczone)") %>%
  kable_styling(bootstrap_options = c("striped", "hover", "condensed"), full_width = F)

## ----------------------------------------------------------
## NAJLEPSZY MODEL NA PODSTAWIE RMSE I MAPE
## ----------------------------------------------------------
# Komentarz tekstowy (opcjonalnie do raportu)
rmse_best <- ifelse(accuracy_arima["Test set", "RMSE"] < accuracy_hw["Test set", "RMSE"], "ARIMA", "Holt-Winters")
mape_best <- ifelse(accuracy_arima["Test set", "MAPE"] < accuracy_hw["Test set", "MAPE"], "ARIMA", "Holt-Winters")

cat(paste("Najlepszy model według RMSE to:", rmse_best))
cat(paste("\nNajlepszy model według MAPE to:", mape_best))


## ----------------------------------------------------------
## WIZUALIZACJA PORÓWNANIA RMSE I MAPE
## ----------------------------------------------------------
# Przygotowanie danych do wykresu słupkowego
porownanie_long <- melt(porownanie, id.vars = "Model", variable.name = "Metryka", value.name = "Wartość")

# Wykres
ggplot(porownanie_long, aes(x = Model, y = Wartość, fill = Model)) +
  geom_bar(stat = "identity", position = position_dodge()) +
  facet_wrap(~Metryka, scales = "free_y") +
  labs(title = "Porównanie modeli prognozujących", y = "Wartość", x = "Model") +
  theme_minimal() +
  theme(legend.position = "none")

