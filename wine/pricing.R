remove_outliers <- function(x, na.rm = TRUE, ...) {
  qnt <- quantile(x, probs=c(.25, .75), na.rm = na.rm, ...)
  H <- 1.5 * IQR(x, na.rm = na.rm)
  y <- x
  y[x < (qnt[1] - H)] <- NA
  y[x > (qnt[2] + H)] <- NA
  y
}

prices1 <- read_csv("data/wine-reviews/winemag-data_first150k.csv")
prices2 <- read_csv("data/wine-reviews/winemag-data-130k-v2.csv")

prices1 <- prices1 %>% drop_na()
prices2 <- prices2 %>% drop_na()

prices <- rbind(prices1, prices2)

plot(density(prices1$price))

port <- filter(prices1, country == "Portugal")
nrow(port)

port2 <- filter(port, province == "Vinho Verde")

remove_outliers(port2$price)

port2 <- port2[port2$price > quantile(port2$price, .25, na.rm = TRUE) - 1.5*IQR(port2$price, na.rm = TRUE) & 
        port2$price < quantile(port2$price, .75, na.rm = TRUE) + 1.5*IQR(port2$price, na.rm = TRUE), ] #rows

# bx <-BoxCoxTrans(port2$points, na.rm = TRUE)
# trans <- bx$lambda
# port2$points <- (port2$points^(trans-1))/(trans)


summary(port2$price)
port2$cut <- cut(port2$price, breaks = 3, labels = FALSE)
port2$cut <- port2$cut + 1

port3 <- port2 %>% 
  group_by(cut) %>% 
  summarise(mean_price = mean(price, na.rm = TRUE),
            se = sd(price, na.rm = TRUE)/sqrt(n()))

port3$min <- port3$mean_price - port3$se*1.96
port3$max <- port3$mean_price + port3$se*1.96
port3

ggplot(port3, aes(x = cut, y = mean_price)) + 
  geom_point() + geom_errorbar(aes(ymin = min, ymax = max), width = .2) +
  theme_tufte(11) +
  annotate("segment", x=-Inf, xend=Inf, y=-Inf, yend=-Inf, color = "grey") +
  annotate("segment", x=-Inf, xend=-Inf, y=-Inf, yend=Inf, color = "grey") +
  ylab("Price ($)") +
  xlab("Rating") +
  scale_x_continuous(breaks = seq(2, 8, 1)) +
  scale_y_continuous(breaks = seq(0, 20, 1))

ggplot(NULL) + geom_density(aes(port2$points)) +
  theme_tufte(11) +
  annotate("segment", x=-Inf, xend=Inf, y=-Inf, yend=-Inf, color = "grey") +
  annotate("segment", x=-Inf, xend=-Inf, y=-Inf, yend=Inf, color = "grey") +
  ylab("Density") +
  xlab("Rating")
ggsave("figures/density_vinho_verde_prices.pdf", width = 6, height = 4)

port3 <- port2 %>% 
  group_by(cut) %>% 
  summarise(mean_price = mean(price, na.rm = TRUE),
            se = sd(price, na.rm = TRUE)/sqrt(n()))

port3$min <- port3$mean_price - port3$se*1.96
port3$max <- port3$mean_price + port3$se*1.96
port3


port3


geom_histogram(aes(dat$quality))

ggplot(NULL) + 
  theme_tufte(11) +
  geom_histogram(data = red, aes(x=as.numeric(as.character(quality)),y=..density..), position="identity") + 
  geom_density(data = port2, aes(x=log(3 + price),y=..density..)) +
  annotate("segment", x=-Inf, xend=Inf, y=-Inf, yend=-Inf, color = "grey") +
  annotate("segment", x=-Inf, xend=-Inf, y=-Inf, yend=Inf, color = "grey") 
