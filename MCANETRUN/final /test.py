import pandas as pd
import mysql.connector
from mysql.connector import Error

sequence = "PFWKILNPLLERGTYYYFMGQQPGKVLGDQRRPSLPALHFIKGAGKKESSRHGGPHCNVFVEHEALQRPVASDFEPQGLSEAARWNSKENLLAGPSENDPNLFVALYDFVASGDNTLSITKGEKLRVLGYNHNGEWCEAQTKNGQGWVPSNYITPVNSLEKHSWYHGPVSRNAAEYLLSSGINGSFLVRESESSPGQRSISLRYEGRVYHYRINTASDGKLYVSSESRFNTLAELVHHHSTVADGLITTLHYPAPKRNKPTVYGVSPNYDKWEMERTDITMKHKLGGGQYGEVYEGVWKKYSLTVAVKTLKEDTMEVEEFLKEAAVMKEIKHPNLVQLLGVCTREPPFYIITEFMTYGNLLDYLRECNRQEVNAVVLLYMATQISSAMEYLEKKNFIHRDLAARNCLVGENHLVKVADFGLSRLMTGDTYTAHAGAKFPIKWTAPESLAYNKFSIKSDVWAFGVLLWEIATYGMSPYPGIDLSQVYELLEKDYRMERPEGCPEKVYELMRACWQWNPSDRPSFAEIHQAFETMFQESSISDEVEKELGKQGVRGAVSTLLQAPELPTKTRTSRRAAEHRDTTDVPEMPHSKGQGESDPLDHEPAVSPLLPRKERGPPEGGLNEDERLLPKDKKTNLFSALIKKKKKTAPTPPKRSSSFREMDGQPERRGAGEEEGRDISNGALAFTPLDTADPAKSPKPSNGAGVPNGALRESGGSGFRSPHLWKKSSTLTSSRLATGEEEGGGSSSKRFLRSCSASCVPHGAKDTEWRSVTLPRDLQSTGRQFDSSTFGGHKSEKPALPRKRAGENRSDQVTRGTVTPPPRLVKKNEEAADEVFKDIMESSPGSSPPNLTPKPLRRQVTVAPASGLPHKEEAGKGSALGTPAAAEPVTPTSKAGSGAPGGTSKGPAEESRVRRHKHSSESPGRDKGKLSRLKPAPPPPPAASAGKAGGKPSQSPSQEAAGEAVLGAKTKATSLVDAVNSDAAKPSQPGEGLKKPVLPATPKPQSAKPSGTPISPAPVPSTLPSASSALAGDQPSSTAFIPLISTRVSLRKTRQPPERIASGAITKGVVLDSTEALCLAISRNSEQMASHSAVLEAGKNLYTFCVSYVDSIQQMRNKFAFREAINKLENNLRELQICPATAGSGPAATQDFSKLLSSVKEISDIVQR"

db_config = {
    "host": "localhost",
    "user": "root",
    "password": "root",
    "database": "dti"
}


def get_db_connection():
    return mysql.connector.connect(**db_config)


try:
    df = pd.read_csv("/Users/renhonglow/PycharmProjects/FinalYearProject/MCANETRUN/data/DavisNKiba.csv")
    matched_row = df[df['Protein Sequence'] == sequence]

    if not matched_row.empty:
        protein_name = matched_row.iloc[0]['Protein Name']
        conn = get_db_connection()
        if conn.is_connected():
            cursor = conn.cursor(dictionary=True)
            query = "SELECT * FROM protein WHERE gene_names = %s"
            cursor.execute(query, (protein_name,))
            result = cursor.fetchone()
            cursor.close()
            conn.close()
            if result:
                print("Database Result:", result)
            else:
                print("No matching record found in the database.")
    else:
        print("No matching row found in the CSV file.")
except Error as e:
    print(f"数据库错误: {e}")
except KeyError as e:
    print(f"KeyError: {e}. 请检查列名是否正确。")
finally:
    try:
        if conn and conn.is_connected():
            cursor.close()
            conn.close()
    except NameError:
        pass