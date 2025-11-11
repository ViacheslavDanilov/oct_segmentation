import base64
import os
from io import BytesIO

from dotenv import load_dotenv
from gigachat import GigaChat
from PIL import Image

load_dotenv()

GIGA_CHAT_TOKEN = os.getenv("GIGA_CHAT_TOKEN")


class GigaChatModel:
    """Класс для работы с моделью GigaChat."""

    def __init__(self, auth_token):
        self.auth_token = auth_token
        self.giga = GigaChat(
            credentials=auth_token,
            model="GigaChat-2-Pro",
            verify_ssl_certs=False,
        )
        response = self.giga.get_token()
        self.access_token = response.access_token

    def get_img_chat_message(self, prompt: str, image: Image.Image = None):
        # image = self.get_pil_to_base64(image)

        # image.save('temp.jpg')
        # file = self.giga.upload_file(image.tobytes())
        # file = self.giga.upload_file(open('temp.jpg', 'rb'))
        # print(file)

        if image is not None:
            img_byte_arr = BytesIO()
            image.save(img_byte_arr, format="JPEG")
            img_byte_arr.seek(0)

        # headers = {
        #     'Authorization': f'Bearer {self.access_token}'
        # }
        #
        # files = {
        #     "file": ("image.jpeg", img_byte_arr, "image/jpeg"),
        #     "purpose": (None, "general")
        # }
        #
        # response = requests.request(
        #     "POST",
        #     f"https://gigachat.devices.sberbank.ru/api/v1/files",
        #     headers=headers,
        #     files=files,
        #     verify=False
        # )
        # response = response.json()

        # print(response.json())

        response = self.giga.chat(
            {
                "function_call": "auto",
                "messages": [
                    {
                        "role": "system",
                        "content": "Вы работаете как система анализа медицинских изображений и биометрических данных, "
                        "специально предназначенная для формирования заключения по результатам оптической "
                        "когерентной томографии (ОКТ). Ваша задача заключается в выявлении потенциальных "
                        "рисков разрыва атеросклеротических бляшек путем оценки определенных показателей, "
                        "включая среднюю площадь сосудистого просвета, количество, толщину и характеристики "
                        "фиброзных покрышек. Цель вашей работы — предоставление клиницистам четкого и "
                        "объективного заключения относительно состояния сосудов пациента и "
                        "риска развития сердечно-сосудистых осложнений."
                        ""
                        "Не нужно использовать водные слова, только четкий и понятный ответ. "
                        "Также не нужно добавлять рассуждения, что данных недостоточно, "
                        "не могу сделать точный ответ и т.п. Пиши врачебные факты и наблюдения"
                        ""
                        "Опирайся на пример:"
                        "Средняя площадь сосудистого просвета составляет 30 ± 4 мкм², что может указывать на наличие стеноза."
                        "Фиброзная покрышка имеет минимальное значение толщины всего 0.19 мкм, что является очень низким показателем и указывает на высокий риск разрыва атеросклеротической бляшки."
                        "Средняя толщина фиброзной покрышки составляет 0.79 ± 0.21 мкм, при этом отмечается высокая вариабельность значений (стандартное отклонение 0.21 мкм)."
                        "Общее количество объектов (предположительно, атеросклеротические бляшки) равно пяти, что также может свидетельствовать о распространенности процесса."
                        "Данные ОКТ указывают на повышенный риск разрыва атеросклеротических бляшек вследствие тонкой и нестабильной фиброзной покрышки."
                        " Рекомендуется консультация кардиолога или сосудистого хирурга для определения дальнейшей тактики ведения пациента, "
                        "которая может включать медикаментозную терапию, изменение образа жизни и, возможно, инвазивные вмешательства."
                        ""
                        "Ответ строго в формате json с полями:"
                        "description: str(описание врачебного заключения)"
                        "risk_classification: str(low, medium, height)",
                    },
                    {
                        "role": "user",
                        "content": prompt,
                    },
                ],
                "temperature": 0.64,
            },
        )
        return response

    @staticmethod
    def get_pil_to_base64(image: Image.Image) -> str:
        return base64.b64encode(image.tobytes()).decode("utf-8")


gigachat = GigaChatModel(
    auth_token=GIGA_CHAT_TOKEN,
)
