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
library(purrr)

## ----------------------------------------------------------
## WCZYTANIE I PRZYGOTOWANIE DANYCH
## ----------------------------------------------------------
url_revenue <- "https://raw.githubusercontent.com/Julia6400/Projekt_Rstudio_szeregiczasowe/main/dane/Walmart_Quarterly_Revenue.csv"
url_profit  <- "https://raw.githubusercontent.com/Julia6400/Projekt_Rstudio_szeregiczasowe/main/dane/Walmart_Quarterly_Gross_Profit.csv"
url_shares  <- "https://raw.githubusercontent.com/Julia6400/Projekt_Rstudio_szeregiczasowe/main/dane/Walmart_Quarterly_Shares_Outstanding.csv"

df_revenue <- read.csv(url_revenue, stringsAsFactors = FALSE)
df_profit  <- read.csv(url_profit, stringsAsFactors = FALSE)
df_shares  <- read.csv(url_shares, stringsAsFactors = FALSE)

# Konwersja dat
format_try <- "%m/%d/%y"
df_revenue$Date <- as.Date(df_revenue$Date, format = format_try)
df_profit$Date  <- as.Date(df_profit$Date, format = format_try)
df_shares$Date  <- as.Date(df_shares$Date, format = format_try)

# Połączenie danych
full_df <- df_revenue %>%
  inner_join(df_profit, by = "Date") %>%
  inner_join(df_shares, by = "Date")

# Wyszukiwanie kolumn
revenue_col <- grep("Revenue", names(full_df), value = TRUE)
profit_col  <- grep("Gross", names(full_df), value = TRUE)
shares_col  <- grep("Shares", names(full_df), value = TRUE)

# Konwersja danych
full_df[[revenue_col]] <- as.numeric(gsub(",", "", full_df[[revenue_col]]))
full_df[[profit_col]]  <- as.numeric(gsub(",", "", full_df[[profit_col]]))
full_df[[shares_col]]  <- as.numeric(gsub(",", "", full_df[[shares_col]]))

# Dodanie kolumn pochodnych
full_df <- full_df %>%
  arrange(Date) %>%
  mutate(
    Kwartał = paste0(year(Date), " Q", quarter(Date)),
    MarzaBrutto = .data[[profit_col]] / .data[[revenue_col]],
    PrzychodNaAkcje = .data[[revenue_col]] / .data[[shares_col]]
  )


## ----------------------------------------------------------
## WIZUALIZACJE DANYCH
## ----------------------------------------------------------
# Wykres przychodu
plot(ts(full_df[[revenue_col]], start = c(2009,1), frequency = 4),
     main = "Szereg czasowy przychodu Walmart (do Q2 2025)",
     ylab = "Przychód (mln USD)", xlab = "Rok", lwd = 2)

# Wykres zysku brutto
ggplot(full_df, aes(x = Date, y = .data[[profit_col]])) +
  geom_line(color = "#2E8B57", linewidth = 1.2) +
  geom_point(color = "#004d00", size = 1) +
  labs(title = "Zysk brutto Walmart", subtitle = "Wartości kwartalne w mln USD",
       x = "Data", y = "Zysk brutto (mln USD)") +
  theme_minimal(base_size = 13) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

# Wykres marży brutto
ggplot(full_df, aes(x = Date, y = MarzaBrutto)) +
  geom_line(color = "#b22222", linewidth = 1.2) +
  geom_point(color = "#800000", size = 1) +
  labs(title = "Marża brutto Walmart", subtitle = "Stosunek zysku brutto do przychodu",
       x = "Data", y = "Marża brutto (%)") +
  scale_y_continuous(labels = scales::percent_format(accuracy = 1)) +
  theme_minimal(base_size = 13) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

# Zależność między marżą a przychodem
ggplot(full_df, aes(x = MarzaBrutto, y = .data[[revenue_col]])) +
  geom_point(color = "#4682b4", size = 2) +
  geom_smooth(method = "lm", se = FALSE, color = "black") +
  labs(title = "Czy wyższa marża brutto oznacza wyższy przychód?",
       x = "Marża brutto (%)", y = "Przychód (mln USD)") +
  scale_x_continuous(labels = scales::percent_format(accuracy = 1)) +
  theme_minimal(base_size = 13)




## ----------------------------------------------------------
## MODELE I PROGNOZY: ARIMA, HOLT-WINTERS, REGRESJA
## ----------------------------------------------------------
revenue_ts <- ts(full_df[[revenue_col]], start = c(2009, 1), frequency = 4)
h_forecast <- 6
train_ts <- revenue_ts

model_arima <- auto.arima(train_ts)
forecast_arima <- forecast(model_arima, h = h_forecast)

model_hw <- ets(train_ts, model = "ZZZ")
forecast_hw <- forecast(model_hw, h = h_forecast)

model_lm <- lm(.data[[revenue_col]] ~ MarzaBrutto + PrzychodNaAkcje, data = full_df)
full_df$Forecast_LM <- predict(model_lm)



## ----------------------------------------------------------
## WIZUALIZACJA PROGNOZ: ARIMA i HOLT-WINTERS
## ----------------------------------------------------------
plot(forecast_arima, xlab = "Rok", ylab = "Przychód (mln USD)",
     main = "Prognoza przychodu Walmart – model ARIMA", col = "blue", lwd = 2)
lines(revenue_ts, col = "black", lwd = 2)
abline(v = time(revenue_ts)[length(revenue_ts)], col = "red", lty = 2)
legend("topleft", legend = c("Rzeczywisty", "Prognoza"), col = c("black", "blue"), lwd = 2, bty = "n")

plot(forecast_hw, xlab = "Rok", ylab = "Przychód (mln USD)",
     main = "Prognoza przychodu Walmart – model Holt-Winters", col = "blue", lwd = 2)
lines(revenue_ts, col = "black", lwd = 2)
abline(v = time(revenue_ts)[length(revenue_ts)], col = "red", lty = 2)
legend("topleft", legend = c("Rzeczywisty", "Prognoza"), col = c("black", "blue"), lwd = 2, bty = "n")



## ----------------------------------------------------------
## METRYKI RMSE i MAPE
## ----------------------------------------------------------
rmse <- function(actual, predicted) sqrt(mean((actual - predicted)^2, na.rm = TRUE))
mape <- function(actual, predicted) mean(abs((actual - predicted) / actual), na.rm = TRUE) * 100

rmse_lm <- rmse(full_df[[revenue_col]], full_df$Forecast_LM)
mape_lm <- mape(full_df[[revenue_col]], full_df$Forecast_LM)

accuracy_arima <- accuracy(forecast_arima, train_ts)
accuracy_hw <- accuracy(forecast_hw, train_ts)

porownanie <- data.frame(
  Model = c("ARIMA", "Holt-Winters", "Regresja"),
  RMSE = round(c(accuracy_arima["Training set", "RMSE"],
                 accuracy_hw["Training set", "RMSE"],
                 rmse_lm), 2),
  MAPE = round(c(accuracy_arima["Training set", "MAPE"],
                 accuracy_hw["Training set", "MAPE"],
                 mape_lm), 2)
)


## ----------------------------------------------------------
## WIZUALIZACJA: RZECZYWISTY PRZYCHÓD VS PROGNOZA Z REGRESJI
## ----------------------------------------------------------
ggplot(full_df, aes(x = Date)) +
  geom_line(aes(y = .data[[revenue_col]], color = "Rzeczywisty"), linewidth = 1.1) +
  geom_line(aes(y = Forecast_LM, color = "Regresja"), linewidth = 1.1, linetype = "dashed") +
  labs(title = "Model regresji przychodu Walmart",
       subtitle = "Zmienna objaśniana: przychód (mln USD)",
       y = "Przychód (mln USD)", x = "Data") +
  scale_color_manual(name = "Legenda",
                     values = c("Rzeczywisty" = "black", "Regresja" = "#377eb8")) +
  theme_minimal(base_size = 13) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))


## ----------------------------------------------------------
## WYBÓR NAJLEPSZEGO MODELU I PROGNOZA DO 2026
## ----------------------------------------------------------

# Przygotowanie forecastów dla porównania na wykresie
future_dates <- seq.Date(from = max(df_all$Date) + 90, by = "3 months", length.out = h_forecast)

# Prognoza z regresji
future_df <- tail(df_all, 1)[rep(1, h_forecast), ]
future_df$Date <- future_dates
future_df$GrossMargin <- mean(df_all$GrossMargin, na.rm = TRUE)
future_df$RevenuePerShare <- mean(df_all$RevenuePerShare, na.rm = TRUE)
future_df$Forecast_LM <- predict(model_lm, newdata = future_df)

# Dane do wykresu prognoz
forecast_df <- data.frame(
  Date = future_dates,
  ARIMA = as.numeric(forecast_arima$mean),
  HoltWinters = as.numeric(forecast_hw$mean),
  Regresja = as.numeric(future_df$Forecast_LM)
)

# Przekształcenie do długiego formatu
forecast_long <- reshape2::melt(forecast_df, id.vars = "Date", variable.name = "Model", value.name = "Prognoza")
forecast_long$Model <- factor(forecast_long$Model, levels = c("ARIMA", "HoltWinters", "Regresja"))

# Dane historyczne
historical_df <- data.frame(Date = df_all$Date, Przychod = df_all[[revenue_col]])

# Wykres porównawczy prognoz w jednym widoku
ggplot() +
  geom_line(data = historical_df, aes(x = Date, y = Przychod), color = "gray30", linewidth = 1) +
  geom_line(data = forecast_long, aes(x = Date, y = Prognoza, color = Model), linewidth = 1.2) +
  scale_color_manual(values = c("ARIMA" = "#1b9e77", "HoltWinters" = "#7570b3", "Regresja" = "#d95f02")) +
  labs(title = "Porównanie prognoz przychodu Walmart do końca 2026",
       subtitle = "Dane historyczne i prognozy z trzech modeli: ARIMA, Holt-Winters i Regresja",
       x = "Data", y = "Przychód (mln USD)", color = "Model") +
  theme_minimal(base_size = 13) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))








## ----------------------------------------------------------
## WIZUALIZACJE: WYKRESY MODELI I PORÓWNANIA
## ----------------------------------------------------------
# Wykres porównania RMSE i MAPE
library(tidyr)
porownanie_long <- pivot_longer(porownanie, cols = c("RMSE", "MAPE"), names_to = "Metryka", values_to = "Wartosc")

ggplot(porownanie_long, aes(x = Model, y = Wartosc, fill = Model)) +
  geom_bar(stat = "identity", position = position_dodge()) +
  facet_wrap(~Metryka, scales = "free_y") +
  labs(title = "Porównanie modeli prognozujących", y = "Wartość", x = "Model") +
  theme_minimal() +
  theme(legend.position = "none")























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
walmart.ts <- ts(df_all$Revenue.Millions.of.US., start = c(2009, 1), frequency = 4)

# Zmiana punktu podziału: 16 ostatnich kwartałów jako walidacja
nValid <- 16
nTrain <- length(walmart.ts) - nValid
train.ts <- window(walmart.ts, end = c(2009 + (nTrain - 1) %/% 4, (nTrain - 1) %% 4 + 1))
valid.ts <- window(walmart.ts, start = c(2009 + nTrain %/% 4, nTrain %% 4 + 1))


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


