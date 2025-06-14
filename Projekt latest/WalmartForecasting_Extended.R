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
library(tidyr)

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
## ANALIZA WSTĘPNA I WIZUALIZACJE
## ----------------------------------------------------------
revenue_ts <- ts(full_df[[revenue_col]], start = c(2009,1), frequency = 4)

plot(revenue_ts, main = "Szereg czasowy przychodu Walmart (do Q2 2025)",
     ylab = "Przychód (mln USD)", xlab = "Rok", lwd = 2)

ggplot(full_df, aes(x = Date, y = .data[[profit_col]])) +
  geom_line(color = "#2E8B57", linewidth = 1.2) +
  geom_point(color = "#004d00", size = 1) +
  labs(title = "Zysk brutto Walmart", subtitle = "Wartości kwartalne w mln USD",
       x = "Data", y = "Zysk brutto (mln USD)") +
  theme_minimal()

ggplot(full_df, aes(x = Date, y = MarzaBrutto)) +
  geom_line(color = "#b22222", linewidth = 1.2) +
  geom_point(color = "#800000", size = 1) +
  labs(title = "Marża brutto Walmart", subtitle = "Stosunek zysku brutto do przychodu",
       x = "Data", y = "Marża brutto (%)") +
  scale_y_continuous(labels = scales::percent_format(accuracy = 1)) +
  theme_minimal()

ggplot(full_df, aes(x = MarzaBrutto, y = .data[[revenue_col]])) +
  geom_point(color = "#4682b4", size = 2) +
  geom_smooth(method = "lm", se = FALSE, color = "black") +
  labs(title = "Czy wyższa marża brutto oznacza wyższy przychód?",
       x = "Marża brutto (%)", y = "Przychód (mln USD)") +
  scale_x_continuous(labels = scales::percent_format(accuracy = 1)) +
  theme_minimal()

# Dekompozycja szeregu czasowego
decomposed <- stl(revenue_ts, s.window = "periodic")
autoplot(decomposed) + ggtitle("Dekompozycja STL przychodów Walmart")

# Test stacjonarności
adf_test <- adf.test(revenue_ts)
cat("Test ADF p-value:", adf_test$p.value, "\n")

# Korelacje
cor_matrix <- cor(full_df[, c(revenue_col, "MarzaBrutto", "PrzychodNaAkcje")])
print(cor_matrix)

## ----------------------------------------------------------
## MODELE I PROGNOZY
## ----------------------------------------------------------
h_forecast <- 8
train_ts <- revenue_ts

model_arima <- auto.arima(train_ts)
forecast_arima <- forecast(model_arima, h = h_forecast)

model_hw <- ets(train_ts)
forecast_hw <- forecast(model_hw, h = h_forecast)

model_lm <- lm(full_df[[revenue_col]] ~ MarzaBrutto + PrzychodNaAkcje, data = full_df)
full_df$Forecast_LM <- predict(model_lm)

## ----------------------------------------------------------
## ANALIZA RESZT DLA MODELI
## ----------------------------------------------------------
# Reszty regresji liniowej
resid_lm <- residuals(model_lm)

# Wykres reszt regresji liniowej
par(mfrow = c(2,2))
plot(resid_lm, main = "Reszty modelu regresji liniowej", ylab = "Reszty", type = "o")
acf(resid_lm, main = "ACF reszt modelu regresji liniowej")
hist(resid_lm, main = "Histogram reszt regresji liniowej", xlab = "Reszty", breaks = 10)
qqnorm(resid_lm); qqline(resid_lm, col = "red")

# Testy reszt regresji liniowej
shapiro_test <- shapiro.test(resid_lm)
cat("Shapiro-Wilk test p-value dla reszt regresji:", shapiro_test$p.value, "\n")
box_test <- Box.test(resid_lm, lag = 10, type = "Ljung-Box")
cat("Test Ljunga-Boxa p-value dla reszt regresji:", box_test$p.value, "\n")

# Analiza reszt ARIMA
checkresiduals(model_arima)

# Analiza reszt Holt-Winters
checkresiduals(model_hw)

par(mfrow = c(1,1)) # Przywrócenie domyślnego układu wykresów

## ----------------------------------------------------------
## WIZUALIZACJE PROGNOZ
## ----------------------------------------------------------
plot(forecast_arima, main = "Prognoza przychodu Walmart – model ARIMA",
     xlab = "Rok", ylab = "Przychód (mln USD)", col = "blue", lwd = 2)
lines(revenue_ts, col = "black", lwd = 2)
abline(v = time(revenue_ts)[length(revenue_ts)], col = "red", lty = 2)

plot(forecast_hw, main = "Prognoza przychodu Walmart – model Holt-Winters",
     xlab = "Rok", ylab = "Przychód (mln USD)", col = "blue", lwd = 2)
lines(revenue_ts, col = "black", lwd = 2)
abline(v = time(revenue_ts)[length(revenue_ts)], col = "red", lty = 2)

## ----------------------------------------------------------
## OCENA MODELI I PORÓWNANIE
## ----------------------------------------------------------
rmse <- function(actual, predicted) sqrt(mean((actual - predicted)^2, na.rm = TRUE))
mape <- function(actual, predicted) mean(abs((actual - predicted) / actual), na.rm = TRUE) * 100

rmse_lm <- rmse(full_df[[revenue_col]], full_df$Forecast_LM)
mape_lm <- mape(full_df[[revenue_col]], full_df$Forecast_LM)

accuracy_arima <- accuracy(model_arima)
accuracy_hw    <- accuracy(model_hw)

porownanie <- data.frame(
  Model = c("ARIMA", "Holt-Winters", "Regresja"),
  RMSE = round(c(accuracy_arima["Training set", "RMSE"],
                 accuracy_hw["Training set", "RMSE"],
                 rmse_lm), 2),
  MAPE = round(c(accuracy_arima["Training set", "MAPE"],
                 accuracy_hw["Training set", "MAPE"],
                 mape_lm), 2)
)

porownanie_long <- pivot_longer(porownanie, cols = c("RMSE", "MAPE"),
                                names_to = "Metryka", values_to = "Wartosc")

ggplot(porownanie_long, aes(x = Model, y = Wartosc, fill = Model)) +
  geom_bar(stat = "identity", position = position_dodge()) +
  facet_wrap(~Metryka, scales = "free_y") +
  labs(title = "Porównanie modeli prognozujących", y = "Wartość", x = "Model") +
  theme_minimal() +
  theme(legend.position = "none")
