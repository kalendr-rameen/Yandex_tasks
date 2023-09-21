# Determining_a_prospective_tariff_for_a_telecom_company

Customers of Megaline, a federal cellular operator, are offered two tariff plans: Smart and Ultra. In order to adjust the advertising budget, the commercial department wants to understand which plan brings in the most money.

It is necessary to make a preliminary analysis of tariffs on a small sample of clients. We have at our disposal the data of 500 "Megaline" users: who they are, where they come from, what tariff they use, how many calls and messages each sent in 2018. You need to analyze customer behavior and draw a conclusion about which tariff is better.

```
Tariffs description

  Tariff "Smart"
      Monthly fee: 550 rubles
      Included are 500 minutes of talk time, 50 messages and 15 GB of Internet traffic
      Fee for services beyond the tariff package:
      1 minute of talk time: 3 rubles
      Message: 3 rubles
      1 Mb of Internet traffic: 200 rubles
  Tariff "Ultra"
      Monthly fee: 1950 rubles
      Included are 3000 minutes of talk time, 1000 messages and 30 GB of Internet traffic
      The cost of services over the tariff package:
      1 minute of talk time: 1 ruble
      message: 1 ruble
      1 GB of Internet traffic: 150 rubles

```

```

Data description

  Table users (information about users):
    user_id - unique user ID
    first_name - user name
    last_name - user's last name
    age - user's age (years)
    reg_date - tariff activation date (day, month, year)
    churn_date - date when user discontinued the tariff (if value is missing, then tariff was active at the moment of data uploading)
    city - user's city of residence
    tariff - name of tariff plan
  Table calls (information about calls):
    id - unique number of call
    call_date - date of the call
    duration - duration of the call in minutes
    user_id - identifier of the user who made the call.
  Table messages (information about messages):
    id - message number
    message_date - message date
    user_id - identifier of the user who sent the message
  Table internet (information about internet sessions):
    id - unique session number
    mb_used - amount of the Internet traffic spent during the session (in megabytes)
    session_date - date of the Internet session
    user_id - user ID
  Tariffs table (information about tariffs):
    tariff_name - tariff name
    rub_monthly_fee - monthly subscription fee in rubles
    minutes_included - number of minutes of conversation per month, included in the subscription fee
    messages_included - number of messages per month included in the monthly subscription fee
    mb_per_month_included - amount of the Internet traffic included in the subscription fee (in megabytes)
    rub_per_minute - the cost of a minute of conversation over the tariff package (for example, if the tariff has 100 minutes of conversation per month, then 101 minutes will be charged)
    rub_per_message - cost of sending a message over the tariff package
    rub_per_gb - the cost of an extra gigabyte of Internet traffic over the tariff package (1 gigabyte = 1024 megabytes.
```        


