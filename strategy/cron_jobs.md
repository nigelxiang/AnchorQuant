# Cron 配置示例

## 1. 调仓建议邮件

每天 16:20 运行一次，脚本内部会自动判断：
- 今天是否是最新交易日
- 今天是否是调仓前一个交易日
- 只有满足条件时才发邮件

```cron
20 16 * * 1-5 cd /home/ubuntu/finance/strategy && /usr/bin/python3 advisor.py --cron-mode >> /home/ubuntu/finance/output/advisor_cron.log 2>&1
```

## 2. 月度收益邮件

每天 16:35 运行一次，脚本内部会自动判断：
- 今天是否是本月首个交易日
- 只有满足条件时才发送“最近一个完整自然月”的收益简报

```cron
35 16 * * 1-5 cd /home/ubuntu/finance/strategy && /usr/bin/python3 main.py --monthly-email --no-chart >> /home/ubuntu/finance/output/monthly_email.log 2>&1
```

## 3. 手工强制发送最近一个完整月收益

```bash
cd /home/ubuntu/finance/strategy
python3 main.py --monthly-email --force-monthly-email --no-chart
```

## 4. SMTP 自检

```bash
cd /home/ubuntu/finance/strategy
python3 advisor.py --smtp-test
```