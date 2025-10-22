import aiohttp
import logging
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Dict, Any, List
from .config import Config
from .data_models import PositionSide

logger = logging.getLogger('FuturesBot')

class NotificationError(Exception):
    """Custom exception for notification errors"""
    pass

class BaseNotifier:
    """Base class for all notifiers"""
    def __init__(self, config: Config):
        self.config = config
        self.enabled = False
        
    async def send_message(self, message: str, subject: str = "") -> bool:
        raise NotImplementedError("Notifier must implement send_message method")
    
    async def notify_trade(self, side: PositionSide, symbol: str, size: float, 
                          price: float, reason: str = "") -> bool:
        raise NotImplementedError("Notifier must implement notify_trade method")
    
    async def notify_exit(self, side: PositionSide, symbol: str, pnl: float, 
                         pnl_percent: float, reason: str = "") -> bool:
        raise NotImplementedError("Notifier must implement notify_exit method")
    
    # Other notification methods would be defined here...

class TelegramNotifier(BaseNotifier):
    def __init__(self, config: Config):
        super().__init__(config)
        self.enabled = config.notification.telegram_enabled
        self.bot_token = config.notification.telegram_bot_token
        self.chat_id = config.notification.telegram_chat_id
        
    async def send_message(self, message: str, subject: str = "") -> bool:
        if not self.enabled:
            return False
        
        url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
        payload = {
            'chat_id': self.chat_id,
            'text': message,
            'parse_mode': 'HTML'
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload) as response:
                    if response.status != 200:
                        logger.error(f"Failed to send Telegram message: {response.status}")
                        return False
                    return True
        except Exception as e:
            logger.error(f"Error sending Telegram message: {e}")
            return False
    
    async def notify_trade(self, side: PositionSide, symbol: str, size: float, 
                          price: float, reason: str = "") -> bool:
        message = (
            f"<b>Trade Executed</b>\n"
            f"Symbol: {symbol}\n"
            f"Side: {side.value}\n"
            f"Size: {size:.4f}\n"
            f"Price: ${price:.2f}\n"
            f"Reason: {reason}"
        )
        return await self.send_message(message)
    
    # Other notification methods would be implemented similarly...

class SlackNotifier(BaseNotifier):
    def __init__(self, config: Config):
        super().__init__(config)
        self.enabled = config.notification.slack_enabled
        self.webhook_url = config.notification.slack_webhook_url
        
    async def send_message(self, message: str, subject: str = "") -> bool:
        if not self.enabled:
            return False
        
        payload = {
            'text': message,
            'username': 'Trading Bot',
            'icon_emoji': ':robot_face:'
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(self.webhook_url, json=payload) as response:
                    if response.status != 200:
                        logger.error(f"Failed to send Slack message: {response.status}")
                        return False
                    return True
        except Exception as e:
            logger.error(f"Error sending Slack message: {e}")
            return False
    
    # Other notification methods would be implemented similarly...

class EmailNotifier(BaseNotifier):
    def __init__(self, config: Config):
        super().__init__(config)
        self.enabled = config.notification.email_enabled
        self.sender = config.notification.email_sender
        self.password = config.notification.email_password
        self.recipient = config.notification.email_recipient
        
    async def send_message(self, message: str, subject: str = "Trading Bot Notification") -> bool:
        if not self.enabled:
            return False
        
        try:
            # Create message
            msg = MIMEMultipart()
            msg['From'] = self.sender
            msg['To'] = self.recipient
            msg['Subject'] = subject
            
            # Add body to email
            msg.attach(MIMEText(message, 'plain'))
            
            # Create server
            server = smtplib.SMTP('smtp.gmail.com', 587)
            server.starttls()
            
            # Login
            server.login(self.sender, self.password)
            
            # Send email
            text = msg.as_string()
            server.sendmail(self.sender, self.recipient, text)
            server.quit()
            
            return True
        except Exception as e:
            logger.error(f"Error sending email: {e}")
            return False
    
    # Other notification methods would be implemented similarly...

class MultiNotifier:
    """Composite notifier that sends messages through all enabled notifiers"""
    def __init__(self, config: Config):
        self.notifiers = [
            TelegramNotifier(config),
            SlackNotifier(config),
            EmailNotifier(config)
        ]
        
    async def send_message(self, message: str, subject: str = "") -> bool:
        results = []
        for notifier in self.notifiers:
            if notifier.enabled:
                result = await notifier.send_message(message, subject)
                results.append(result)
        return any(results)
    
    async def notify_trade(self, side: PositionSide, symbol: str, size: float, 
                          price: float, reason: str = "") -> bool:
        results = []
        for notifier in self.notifiers:
            if notifier.enabled:
                result = await notifier.notify_trade(side, symbol, size, price, reason)
                results.append(result)
        return any(results)
    
    # Other notification methods would be implemented similarly...